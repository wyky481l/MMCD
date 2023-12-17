import pytorch_lightning as pl
from torch.nn.functional import mse_loss
from tqdm import tqdm
from modules.sequence.encode import *
from modules.sequence.transformer import SeqTransformer
from modules.structure.egnn import EGNN
from modules.structure.encode import *
from util.constant import seq_length_freq, get_seq_constant_init
from util.diffusion_util import get_para_schedule, clip_norm
from util.embed.embedding import structure_embedding, sequence_embedding
from util.embed.sequence import index_to_fasta
from util.geometry import Peptide


# Multi-Modal Contrastive Diffusion Model
class MMCD(pl.LightningModule):
    def __init__(
            self,
            # parameters for structure diffusion
            struct_node_input_dim=46,
            struct_node_hidden_dim=128,
            struct_edge_dim=8,
            struct_edge_hidden_dim=32,
            struct_node_output_dim=128,
            struct_n_layer=4,
            # layer number for EGNN

            # parameters for sequence diffusion
            seq_n_class=20,
            # residue types
            seq_n_seq_emb=80,
            seq_n_hidden=128,
            seq_clamp=-50,
            seq_n_blocks=8,
            # block number for transformer

            n_timestep=1000,
            n_self_atte_head=4,
            beta_schedule="linear",
            beta_start=1.e-7,
            beta_end=2.e-2,
            temperature=0.1,
            learning_rate_struct=5e-3,
            learning_rate_seq=5e-3,
            learning_rate_cont=5e-3,
            # lr for structure/sequence diffusion and CL
            loss_weight=0.9
            # contributions of diffusion and CL

    ):
        super().__init__()

        self.learning_rate_struct = learning_rate_struct
        self.learning_rate_seq = learning_rate_seq
        self.learning_rate_cont = learning_rate_cont
        self.loss_weight = loss_weight
        self.temperature = temperature
        self.time_sampler = torch.distributions.Categorical(torch.ones(n_timestep))
        self.seq_constant_data = get_seq_constant_init(self.device)

        betas, alphas, alphas_bar = get_para_schedule(
            beta_schedule=beta_schedule,
            beta_start=beta_start,
            beta_end=beta_end,
            num_diffusion_timestep=n_timestep,
            device=self.device
        )

        self.betas = nn.Parameter(betas, requires_grad=False)
        self.alphas = nn.Parameter(alphas, requires_grad=False)
        self.alphas_bar = nn.Parameter(alphas_bar, requires_grad=False)
        self.num_timestep = n_timestep

        # for structure diffusion
        self.edge_attr_mlp = MLPEdgeEncoder(
            edge_dim=struct_edge_dim,
            output_dim=struct_edge_hidden_dim
        )

        self.egnn = EGNN(
            node_input_dim=struct_node_input_dim,
            node_hidden_dim=struct_node_hidden_dim,
            edge_dim=struct_edge_hidden_dim,
            node_output_dim=struct_node_output_dim,
            num_layer=struct_n_layer
        )

        self.struct_ffn = StructFFN(
            input_dim=struct_node_output_dim,
            hidden_dim=struct_node_hidden_dim
        )

        # for sequence diffusion
        self.n_class = seq_n_class
        self.clamp = seq_clamp

        self.transformer = SeqTransformer(
            input_dim=seq_n_seq_emb,
            output_dim=seq_n_hidden,
            n_block=seq_n_blocks
        )

        self.seq_ffn = SeqFFN(seq_n_hidden, seq_n_class)

        # for CL
        self.struct_attention = SelfAttention(
            n_emb=struct_node_output_dim,
            n_head=n_self_atte_head
        )

        self.seq_attention = SelfAttention(
            n_emb=seq_n_hidden,
            n_head=n_self_atte_head
        )

        self.sentence_predictor = MetricPredictorLayer(input_dim=seq_n_hidden)
        self.seq_predictor = MetricPredictorLayer(input_dim=seq_n_hidden)

        self.graph_predictor = MetricPredictorLayer(input_dim=struct_node_output_dim)
        self.struct_predictor = MetricPredictorLayer(input_dim=struct_node_output_dim)

        self.metric_loss = MetricLoss(temperature=temperature)
        self.match_loss = MatchLoss(temperature=temperature)

    def get_loss(self, batch):
        batch_size = len(batch.x)
        time_step = torch.ones(batch_size, device=self.device, dtype=torch.int64) * self.time_sampler.sample()
        # sample a timestep for mini-batch

        # --------------------------- diffusion process ----------------------------
        # structure diffusion
        pos_noise_pred, amp_graph_cont_emb, amp_graph_match_emb, pos_0, pos_t, pos_noise_t, a_pos = \
            self.struct_forward(batch, time_step, "AMP")

        _, nonamp_graph_cont_emb, *results = \
            self.struct_forward(batch, time_step, "nonAMP")

        # sequence diffusion
        amp_x0_real, amp_x0_pred, amp_sentence_cont_emb, amp_sentence_match_emb = self.seq_forward(
            time_step,
            batch,
            batch_size,
            "AMP")

        _, _, nonamp_sentence_cont_emb, _ = self.seq_forward(time_step, batch, batch_size,
                                                             "nonAMP")

        # structure loss
        struct_pos_loss = mse_loss(pos_noise_pred, pos_noise_t)
        # self.log("structDiff_loss", struct_pos_loss, prog_bar=True)

        pred_pos_0 = (1. / a_pos).sqrt() * (pos_t - (1.0 - a_pos).sqrt() * pos_noise_pred)
        struct_pred_loss = torch.sqrt(mse_loss(pred_pos_0, pos_0))
        self.log("struct_score", struct_pred_loss, prog_bar=True)

        # sequence loss
        seq_kl_loss = multinomial_kl(amp_x0_pred, amp_x0_real)
        # self.log("seqDiff_loss", seq_kl_loss, prog_bar=True)

        seq_pred_score = token_aa_acc(amp_x0_pred, amp_x0_real, self.device)
        self.log("seq_score", seq_pred_score, prog_bar=True)

        diff_loss = struct_pos_loss + seq_kl_loss
        self.log("diff_loss", diff_loss, prog_bar=True)

        # ----------------------- CL process ----------------------
        # IntraCL
        struct_metric_loss = self.metric_loss(amp_graph_cont_emb, nonamp_graph_cont_emb)
        # self.log("struct_metric_loss", struct_metric_loss, prog_bar=True)

        seq_metric_loss = self.metric_loss(amp_sentence_cont_emb, nonamp_sentence_cont_emb)
        # self.log("seq_metric_loss", seq_metric_loss, prog_bar=True)

        intra_loss = struct_metric_loss + seq_metric_loss
        # self.log("intra_loss", metric_loss, prog_bar=True)

        # InterCL
        inter_loss = self.match_loss(amp_graph_match_emb, amp_sentence_match_emb, match_type="graph")
        # self.log("inter_loss", match_loss, prog_bar=True)

        contrast_loss = intra_loss + inter_loss
        self.log("contrast_loss", contrast_loss, prog_bar=True)

        total_loss = self.loss_weight * diff_loss + (1 - self.loss_weight) * contrast_loss
        self.log("total_loss", total_loss, prog_bar=True)

        return total_loss, pred_pos_0, pos_0, amp_x0_pred, amp_x0_real

    def struct_pred(self, x, edge_index, edge_attr, edge_length, pos, batch, time_step):
        edge_attr = self.edge_attr_mlp(
            edge_attr=edge_attr,
            edge_length=edge_length
        )

        node_emb = self.egnn(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            edge_length=edge_length,
            batch=batch,
            time_step=time_step,
            pos=pos
        )

        pos_noise_pred = self.struct_ffn(node_emb, time_step, batch)

        return pos_noise_pred, node_emb

    def struct_forward(self, batch, time_step, struct_type, diff_statue: bool = True):
        assert struct_type in {"AMP", "nonAMP"}, print("struct_type error")

        alphas_bar = self.alphas_bar.index_select(0, time_step)

        if struct_type == "AMP":
            pos = batch.pos
            fasta_list = batch.fasta
        else:
            pos = batch.nonamp_pos
            fasta_list = batch.nonamp_fasta

        batch_index = get_batch_info(fasta_list, self.device)
        a_pos = alphas_bar.index_select(0, batch_index).unsqueeze(-1).unsqueeze(-1)

        pos_noise_t = torch.randn_like(pos, device=self.device)

        pos_t = a_pos.sqrt() * pos + pos_noise_t * (1.0 - a_pos).sqrt()

        node_emb, edge_index, edge_attr, edge_length = get_batch_structure_embedding(pos_t, batch_index, fasta_list,
                                                                                     self.device,
                                                                                     self.seq_constant_data)

        pos_noise_pred, node_emb = self.struct_pred(node_emb, edge_index, edge_attr, edge_length, pos_t,
                                                    batch_index,
                                                    time_step)

        pos_noise_pred = clip_norm(pos_noise_pred)
        pos_noise_pred = pos_noise_pred.reshape(-1, 4, 3)

        if diff_statue:
            node_emb, attn = self.struct_attention(node_emb, batch=batch_index)
            graph_emb = get_attn_emb(node_emb, attn, batch_index)
            graph_emb = torch.concat(graph_emb, dim=0)

            graph_cont_emb = self.struct_predictor(graph_emb)
            graph_match_emb = self.graph_predictor(graph_emb)
        else:
            graph_cont_emb = None
            graph_match_emb = None

        return pos_noise_pred, graph_cont_emb, graph_match_emb, pos, pos_t, pos_noise_t, a_pos

    def seq_pred(self, seq_data, time_step, batch):
        seq_emb = self.transformer(seq_data, time_step, batch)
        output = self.seq_ffn(seq_emb)

        seq_pred = F.softmax(output, dim=-1).float()

        return seq_pred, seq_emb

    def seq_forward(self, seq_time_steps, batch, batch_size, seq_type, diff_statue: bool = True):
        assert seq_type in {"AMP", "nonAMP"}, print("seq_type error")

        if seq_type == "AMP":
            x0_real = batch.logit
            batch_index = get_batch_info(batch.fasta, self.device)
        else:
            x0_real = batch.nonamp_logit
            batch_index = get_batch_info(batch.nonamp_fasta, self.device)

        token_time_steps = seq_time_steps.index_select(0, batch_index)
        alphas_bar = self.alphas_bar.index_select(0, seq_time_steps)

        noise = get_seq_noise(device=self.device)

        Qt_weight = get_Qt_weight(alphas_bar, noise, batch_index, self.device, self.n_class)
        x_t = torch.matmul(x0_real.unsqueeze(1), Qt_weight).reshape(-1, self.n_class)

        x_t_emb = batch_sequence_embedding(x_t, batch_index, batch_size, self.device)
        x0_pred, token_emb = self.seq_pred(x_t_emb, token_time_steps, batch_index)

        if diff_statue:
            token_emb, attn = self.seq_attention(token_emb, batch=batch_index)
            sentence_emb = get_attn_emb(token_emb, attn, batch_index)
            sentence_emb = torch.concat(sentence_emb, dim=0)

            sentence_cont_emb = self.seq_predictor(sentence_emb)
            sentence_match_emb = self.sentence_predictor(sentence_emb)
        else:
            sentence_cont_emb = None
            sentence_match_emb = None

        return x0_real, x0_pred, sentence_cont_emb, sentence_match_emb

    def q_posterior(self, x0, time_step, batch):
        """
        p_theta(xt_1|xt) = q(xt-1|xt,x0)*p(x0|xt)

        log(p_theta(xt_1|xt)) = log(q(xt-1|xt,x0)) + log(p(x0|xt))
                               = log(p(x0|xt)) + log(q(xt|xt-1,x0)) + log(q(xt-1|x0)) - log(q(xt|x0))

        Bayesian Rule: log(q(xt-1|xt,x0)) -> log(q(xt|xt-1,x0)) + log(q(xt-1|x0)) - log(q(xt|x0))
        """

        time_step = (time_step + (self.num_timestep + 1)) % (self.num_timestep + 1)

        alphas = self.alphas.index_select(0, time_step)
        alphas_bar_t = self.alphas_bar.index_select(0, time_step)
        alphas_bar_t_1 = self.alphas_bar.index_select(0, time_step - 1)

        noise = get_seq_noise(device=self.device)
        # with marginal distribution

        # q(xt|x0)
        # xt_from_x0 = token_alphas_bar_t.sqrt() * x0 + noise * (1.0 - token_alphas_bar_t).sqrt()
        Qt_weight = get_Qt_weight(alphas_bar_t, noise, batch, self.device, self.n_class)
        xt_from_x0 = torch.matmul(x0.unsqueeze(1), Qt_weight).reshape(-1, self.n_class)

        # q(xt|xt_1,x0) -> q(xt|xt_1)
        # xt_from_xt_1 = token_alphas.sqrt() * x0 + (1 - token_alphas).sqrt() * noise
        Qt_weight = get_Qt_weight(alphas, noise, batch, self.device, self.n_class)
        xt_from_xt_1 = torch.matmul(x0.unsqueeze(1), Qt_weight).reshape(-1, self.n_class)

        # q(xt-1|x0)
        # xt_1_from_x0 = token_alphas_bar_t_1.sqrt() * x0 + noise * (1.0 - token_alphas_bar_t_1).sqrt()
        Qt_weight = get_Qt_weight(alphas_bar_t_1, noise, batch, self.device, self.n_class)
        xt_1_from_x0 = torch.matmul(x0.unsqueeze(1), Qt_weight).reshape(-1, self.n_class)

        # p(x0|xt)
        # x_part = torch.log(x0) - torch.log(xt_from_x0)
        # log(p(x0|xt))-log(q(xt|x0))=log(p(x0|xt)/q(xt|x0))
        # x_log_sum_exp = torch.logsumexp(x_part, dim=-1, keepdim=True)
        # x_part = x_part - x_log_sum_exp

        xt_1_from_xt = torch.log(x0) - torch.log(xt_from_x0) + torch.log(xt_from_xt_1) + torch.log(xt_1_from_x0)
        xt_1_from_xt = torch.clamp(xt_1_from_xt, self.clamp, 0)
        # log(p_theta(xt_1|xt))=log(p(x0|xt)) - log(q(xt|x0)) + log(q(xt|xt-1,x0)) + log(q(xt-1|x0))
        xt_1_from_xt = torch.exp(xt_1_from_xt)
        # p_theta(xt_1|xt)

        return xt_1_from_xt

    @torch.no_grad()
    def denoise_struct_sample(self, fasta_seq, pdb_out_statue: bool = False, pdb_name=None):

        pos_init = torch.randn((len(fasta_seq), 4, 3), device=self.device)

        n_steps = self.num_timestep
        pos = pos_init
        pos_traj = []
        t_list = torch.arange(n_steps - 1, 0, -1).to(self.device)

        for i in tqdm(range(len(t_list))):
            node_embedding, edge_index, edge_attr, edge_length = structure_embedding(pos,
                                                                                     fasta=fasta_seq,
                                                                                     constant_data=self.seq_constant_data)

            pos_noise, _ = self.struct_pred(node_embedding, edge_index, edge_attr, edge_length, pos,
                                            batch=None, time_step=t_list[i])
            pos_noise = pos_noise.reshape(-1, 4, 3)

            e = pos_noise
            x_t = pos
            alpha_bar_t = self.alphas_bar[t_list[i]]
            alpha_t = self.alphas[t_list[i]]
            beta_t = self.betas[t_list[i]]

            mean_eps = (1. / alpha_t).sqrt() * (x_t - beta_t * e / (1. - alpha_bar_t).sqrt())

            log_var = beta_t.log()
            noise = torch.randn_like(mean_eps, device=self.device)
            pos_t_1 = mean_eps + torch.exp(0.5 * log_var) * noise

            pos = pos_t_1
            pos_traj.append(pos[:, 0, :].clone().cpu())

        if pdb_out_statue:
            peptide = Peptide(pos, fasta_seq)
            peptide.reconstruct()
            if pdb_name is None:
                pdb_name = fasta_seq[:5]
            peptide.output_to_pdb(pdb_name)

        return pos, pos_traj

    @torch.no_grad()
    def denoise_seq_sample(self, n_seq=1, seq_length=None, fasta_out_statue: bool = False):
        seq_freq = torch.tensor(seq_length_freq, device=self.device)
        D = torch.distributions.Categorical(seq_freq)

        out_seq_list = []
        out_seq_traj = []

        for i in range(n_seq):
            if seq_length is None:
                seq_len = D.sample()
            else:
                seq_len = seq_length[i]
                # seq_len = seq_length

            seq_init = get_seq_noise(seq_len, self.device)
            seq_index_t = logit_to_index(seq_init, random_state=True)

            batch = torch.zeros(seq_len, device=self.device).long()

            t_list = torch.arange(self.num_timestep - 1, 0, -1).to(self.device)
            print("denoise {}-th sequence".format(i + 1))

            for time_steps in tqdm(t_list):
                seq_emb = sequence_embedding(index=seq_index_t)

                seq_emb = torch.tensor(seq_emb, device=self.device).float()
                token_time_steps = time_steps.repeat(seq_len)

                seq0_pred, _ = self.seq_pred(seq_emb, token_time_steps, batch)
                seq_t = self.q_posterior(seq0_pred, token_time_steps, batch)
                seq_index_t = logit_to_index(seq_t, random_state=True)

                out_seq_traj.append(index_to_fasta(seq_index_t))

            seq_index_final = seq_index_t
            seq_fasta = index_to_fasta(seq_index_final)
            out_seq_list.append(seq_fasta)

        if fasta_out_statue:
            record_path = save_output_seq(out_seq_list)
        else:
            record_path = None

        return out_seq_list, out_seq_traj, record_path

    def training_step(self, batch, batch_idx):
        total_loss, pred_pos_0, pos_0, amp_x0_pred, amp_x0_real = self.get_loss(
            batch=batch,
        )

        self.log("train/loss", total_loss)

        return {"loss": total_loss, "pred_pos_0": pred_pos_0, "pos_0": pos_0, "amp_x0_pred": amp_x0_pred,
                "amp_x0_real": amp_x0_real}

    def training_epoch_end(self, training_step_outputs):
        epoch_pred_pos_0 = torch.concat([step["pred_pos_0"] for step in training_step_outputs], dim=0)
        epoch_pos_0 = torch.concat([step["pos_0"] for step in training_step_outputs], dim=0)

        epoch_struct_pred_loss = torch.sqrt(mse_loss(epoch_pred_pos_0, epoch_pos_0))
        self.log("total_struct_loss", epoch_struct_pred_loss, prog_bar=True)

        epoch_amp_x0_pred = torch.concat([step["amp_x0_pred"] for step in training_step_outputs], dim=0)
        epoch_amp_x0_real = torch.concat([step["amp_x0_real"] for step in training_step_outputs], dim=0)
        epoch_seq_pred_score = token_aa_acc(epoch_amp_x0_pred, epoch_amp_x0_real, self.device)

        self.log("total_seq_score", epoch_seq_pred_score, prog_bar=True)

    def configure_optimizers(self):

        optimizer = torch.optim.Adam([
            # structure
            {'params': self.edge_attr_mlp.parameters(), 'lr': self.learning_rate_struct},
            {'params': self.egnn.parameters(), 'lr': self.learning_rate_struct},
            {'params': self.struct_ffn.parameters(), 'lr': self.learning_rate_struct},

            # sequence
            {'params': self.transformer.parameters(), 'lr': self.learning_rate_seq},
            {'params': self.seq_ffn.parameters(), 'lr': self.learning_rate_seq},

            # CL
            {'params': self.struct_attention.parameters(), 'lr': self.learning_rate_cont},
            {'params': self.seq_attention.parameters(), 'lr': self.learning_rate_cont},
            {'params': self.sentence_predictor.parameters(), 'lr': self.learning_rate_cont},
            {'params': self.graph_predictor.parameters(), 'lr': self.learning_rate_cont},
            {'params': self.seq_predictor.parameters(), 'lr': self.learning_rate_cont},
            {'params': self.struct_predictor.parameters(), 'lr': self.learning_rate_cont},
        ])

        return optimizer

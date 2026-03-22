        """
        输入：
        - input_ids: 原始 token id 序列
        - input_embeddings: 原始 embedding 序列
        - loss_mask_full: 全局可 mask 位掩码（包括动作 token 位置）
        - mask_token_id: 用于填充的 special mask token id
        - no_mask_token_prob: 可选概率，把已 mask 掉的位置再随机 unmask
        返回：
        - masked_input_ids: 用 mask_token_id 替代被 mask 掉的位置的 input_ids
        - labels: 原 input_ids，在非被 mask 位置用 -100 屏蔽（CrossEntropyLoss 忽略）
        - new_input_embeddings: 对应替换了 action token 的新 embeddings
        - loss_mask: float mask，用于后续 loss 加权（1 表示预测该位置，0 表示忽略）
        """
        B, L = input_ids.shape
        if debug_logging:
            print(f"Input shape: {input_ids.shape}")
            print(f"loss_mask_full shape: {loss_mask_full.shape}")
            print(f"loss_mask_full[0]: {loss_mask_full[0]}")
            print(f"loss_mask_full[1]: {loss_mask_full[1]}")
        device = input_ids.device

        # 1) 计算每个样本总共可 mask 的 token 数量
        #    total_unknown = loss_mask_full.sum(dim=1)  # (B,)
        total_unknown = loss_mask_full.float().sum(dim=1)  # (B,)
        if debug_logging:
            print(f"Total unknown tokens per sample: {total_unknown}")

        # 2) 随机采一个 time ratio in [0,1)
        rand_time = torch.rand(B, device=device)
        if debug_logging:
            print(f"Random time ratios: {rand_time}")

        # 3) 根据 schedule 计算 mask ratio、再算出每个样本要 mask 的 token 数
        #    mask_ratios: tensor (B,), 取值 in (0,1]
        mask_ratios = mask_schedule.schedule(rand_time, total_unknown, method="cosine")  # [B]
        if debug_logging:
            print(f"Mask ratios: {mask_ratios}")
        #    num_mask: at least 1
        num_mask = torch.clamp((total_unknown * mask_ratios).round(), min=1).long()  # [B]
        if debug_logging:
            print(f"Number of tokens to mask per sample: {num_mask}")
        # 4) 为每个位置打随机分数，非可-mask 位置打上大数，保证它永远不被选中
        #    vals: (B, L) ~ Uniform(0,1)
        vals = torch.rand(B, L, device=device)
        #    large = 1e8
        large = float('inf')
        #    只有 loss_mask_full==True 的位置保留原分数，其他位置加大
        vals = torch.where(loss_mask_full, vals, vals + large)  # inf 表示不可选

        # 5) 按行排序、取前 num_mask
        perm = vals.argsort(dim=1)                     # (B, L)
        ranks = perm.argsort(dim=1)
        # masked_mask: bool (B, L)，True 表示该位置被 mask 掉
        masked_mask = ranks < num_mask[:, None]        # (B, L), 先 top-k mask

        # 6) 可选：no_mask_token_prob，再次随机取消一部分已 mask 的位置
        if no_mask_token_prob > 0:
            # 从 masked_mask 中随机抽取一部分不再 mask
            # 生成同形状的 [0,1) 随机数
            prob = torch.rand(B, L, device=device)
            # 在已经 mask 的位置上，若 prob < no_mask_token_prob 则 unmask
            unmask = (prob < no_mask_token_prob) & masked_mask
            masked_mask = masked_mask & (~unmask)

        if debug_logging:
            print(f"Masked mask[loss_mask_full][:56] : {masked_mask[loss_mask_full][:56]}")

        if rtc_masking:
            # Replace the first d * ACTION_DIM positions with the original token (False)
            d = np.random.randint(1, int(rtc_masking_H/2) + 1)
            frist_d_mask = torch.ones((B, NUM_ACTIONS_CHUNK, ACTION_DIM), dtype=torch.bool, device=device)
            frist_d_mask[:, :d, :] = False
            # print(f"First d mask[0] : {frist_d_mask[0]}")
            masked_mask[loss_mask_full] = masked_mask[loss_mask_full] & frist_d_mask.reshape(-1)
            # print(f"Masked mask after first d masking[0] : {masked_mask[loss_mask_full][:56]}")

            # Replace the last s * ACTION_DIM positions with the mask token (True)
            # last_s_mask = torch.zeros((B, NUM_ACTIONS_CHUNK, ACTION_DIM), dtype=torch.bool, device=device)
            # last_s_mask[:, -s:, :] = True
            # masked_mask[loss_mask_full] = masked_mask[loss_mask_full] | last_s_mask.reshape(-1)

        if debug_logging:
            print(f"After RTC Masking Masked mask[loss_mask_full][:56] : {masked_mask[loss_mask_full][:56]}")


        # # Set to True in eos_pos
        # if eos_pos is not None:
        #     # eos_pos: (B,), 只在这些位置上 mask 掉
        #     eos_mask = torch.zeros_like(masked_mask, dtype=torch.bool, device=device)
        #     eos_mask[torch.arange(B, device=device), eos_pos] = True
        #     masked_mask = masked_mask | eos_mask

        # 7) 构造 labels: 被 mask 掉的位置保留原 id，其他位置设为 -100
        ignore_labels = torch.full_like(labels, fill_value=IGNORE_INDEX, dtype=labels.dtype, device=device)
        masked_labels = torch.where(masked_mask, labels, ignore_labels)

        masked_input_ids = torch.where(masked_mask, mask_token_id, input_ids)
        # 8) 构造新的 input_embeddings: masked 位置替换成 用 embedding lookup 或直接替换
        #    假设你后面是用 inputs_embeds，所以直接对 embeddings 替换
        #    masked_input_embeddings: (B, L, D)
        masked_input_embeddings = input_embeddings.clone()
        #    获取 mask token embedding
        mask_emb = self.get_input_embeddings()(torch.tensor([mask_token_id], device=device))  # (1, D)
        #    扩展到 (B, L, D)
        mask_emb = mask_emb.view(1, 1, -1).expand(B, L, -1)
        #    替换
        masked_input_embeddings = torch.where(masked_mask.unsqueeze(-1), mask_emb, masked_input_embeddings)

        # 10) 返回
        #     loss_mask: float 跟 JAX 里一致，用于后续加权 loss (1. for masked positions, 0. elsewhere)
        loss_mask = masked_mask.float()

        return masked_input_ids, masked_input_embeddings, masked_labels, loss_mask


def realtime_decode(
    init_ids: torch.LongTensor,             # [B, L], 初始序列（含 mask_token_id）
    tokens_to_logits,                       # fn(seq_ids: [B, L]) -> logits [B, L, V]
    mask_token_id: int,
    start_iter: int = 0,
    num_iter: int = 12,
    choice_temperature: float = 1.0,
    mask_scheduling_method="cosine",
    use_remask: bool = False,              # 是否使用重 mask 概率
    token_critic: torch.nn.Module = None,  # TokenCritic 模型，输出 [B, L] 的分数
    critic_noise_scale: float = 1.0,       # Critic 加噪声比例
    

    inference_mode: Optional[Any] = None,
):
    """
    非自回归 MaskGIT 推理
    返回 final_seqs: [B, num_iter, L]，每轮迭代的 sampled 序列
    """
    B, L = init_ids.shape
    device = init_ids.device

    d_plus_s = inference_mode.d + inference_mode.s


    # 记录初始未知（mask）数量，用于调度
    unknown_init = (init_ids == mask_token_id).sum(dim=1)  # [B]

    # Linear delta determination with unknown_init
    delta = int(num_iter * (1 - unknown_init.float() / L).max().item())
    delta_num_iter = num_iter - delta

    # State init
    cur_seqs = init_ids.clone()                         # [B, L]

    # 迭代解码
    for step in range(start_iter, delta_num_iter):
        # 1) 得到 logits & 概率分布
        logits, actions_hidden_states = tokens_to_logits(cur_seqs)             # [B, L, V]
        probs = F.softmax(logits, dim=-1)               # [B, L, V]

        # 2) 并行 categorical 采样
        #    展平后采样，再 reshape
        flat_probs = probs.view(-1, probs.size(-1))     # [B*L, V]
        sampled_flat = torch.multinomial(flat_probs, 1)  # [B*L, 1]
        sampled = sampled_flat.view(B, L)               # [B, L]

        # 3) 仅在 mask 位置更新
        unknown_map = cur_seqs == mask_token_id         # [B, L]
        sampled = torch.where(unknown_map, sampled, cur_seqs)

        # 4) 计算下轮 mask 数量
        ratio = torch.tensor(float(step + 1) / delta_num_iter, device=device)              # scalar
        # 调度函数：给定 ratio、初始未知数，返回 mask_ratio
        mask_ratio = mask_schedule(ratio, unknown_init, mask_scheduling_method)  # [B]
        mask_len = torch.floor(unknown_init.float() * mask_ratio).long()
        # 保证至少 1 且最多 unknown_init-1
        mask_len = torch.clamp(mask_len, min=1, max=(unknown_init - 1).item())

        # —————————————— 4) 计算每个位置得分 scores ——————————————
        if token_critic is not None:
            # 用 Critic 得分 + 加噪声
            # 假设 token_critic 返回 [B, L] raw scores
            raw_crit = token_critic(actions_hidden_states)              # [B, L]
            scores = - raw_crit
            # 加均匀噪声，看 step 越大噪声越小
            scores = scores + (torch.rand_like(scores).cuda() - 0.5) * critic_noise_scale * (1.0 - ratio)
            selected_probs = scores
        else:
            # 5) 计算每个位置被选中的概率：probs.gather
            selected_probs = probs.gather(2, sampled.unsqueeze(-1)).squeeze(-1)  # [B, L]

        if use_remask:
            # 6) 引入“重 mask 概率”
            #    p_remask 从 1 线性降到 0：早期更容易重新 mask，后期更稳定
            p_remask = 1.0 - ratio
            #    对已知位置（~unknown_map）降低它们的置信度
            selected_probs = torch.where(
                unknown_map,
                selected_probs,
                selected_probs * p_remask
            )  # [B, L]
        else:
            # 已知位置（初始非 mask）设为极大值，避开下轮 mask
            inf = torch.tensor(float("inf"), device=device)
            selected_probs = torch.where(unknown_map, selected_probs, inf)

        # 6) 用 Gumbel+top-k 策略决定下轮仍要被 mask 的位置
        action_mask = mask_by_random_topk(
            selected_probs,
            mask_len,
            temperature=choice_temperature * (1.0 - ratio),
        )

        # 7) 构造 next seqs：被 mask 的位置继续用 mask_token
        next_seqs = torch.where(action_mask, mask_token_id, sampled)  # [B, L]
        cur_seqs = next_seqs

        # 8) Early Stop Mechanism
        # If the first d + s actions are all not masked, we can stop early with discrete_rtc mode
        curr_action_mask = cur_seqs == mask_token_id
        if not curr_action_mask[:, :d_plus_s * ACTION_DIM].any() and inference_mode == "discrete_rtc":
            action_mask = curr_action_mask
            return cur_seqs, action_mask, actions_hidden_states


    # return final_seqs, actions_hidden_states
    # return cur_seqs, action_mask, actions_hidden_states
    return sampled, action_mask, actions_hidden_states
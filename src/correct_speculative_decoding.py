"""
正确的Speculative Decoding实现
演示Edge和Cloud模型之间的正确上下文传递
"""

import torch
import time
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from models.edge_model import EdgeModel
from models.cloud_model import CloudModel

logger = logging.getLogger(__name__)


@dataclass
class CorrectSpeculativeResult:
    """正确speculative decoding的结果"""
    final_text: str
    accepted_tokens: List[int]
    rejected_tokens: List[int]
    total_latency: float
    edge_latency: float
    cloud_latency: float
    acceptance_rate: float
    tokens_per_second: float


class CorrectSpeculativeDecodingSystem:
    """
    正确的Speculative Decoding系统
    演示Edge和Cloud模型之间的正确上下文传递
    """
    
    def __init__(self, edge_model: EdgeModel, cloud_model: CloudModel):
        self.edge_model = edge_model
        self.cloud_model = cloud_model
        logger.info("Initialized Correct Speculative Decoding System")
    
    def generate_with_correct_speculative_decoding(self,
                                                 audio_waveform: torch.Tensor,
                                                 prompt: str = "Based on this audio, describe the emotional state of the speaker in Chinese.",
                                                 block_size: int = 4,
                                                 max_blocks: int = 6) -> CorrectSpeculativeResult:
        """
        正确的Speculative Decoding流程
        
        关键点：
        1. Edge模型生成候选tokens
        2. Cloud模型验证这些候选tokens（接收完整上下文）
        3. 接受/拒绝机制决定最终输出
        """
        total_start_time = time.time()
        
        logger.info("Starting CORRECT speculative decoding process...")
        
        # Step 1: 准备初始上下文（音频+prompt）
        conversation = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio_waveform},
                    {"type": "text", "text": prompt}
                ],
            },
        ]
        
        # 应用chat template
        initial_text = self.edge_model.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        
        # 处理初始输入
        initial_inputs = self.edge_model.processor(
            text=initial_text, 
            audio=audio_waveform,
            return_tensors="pt", 
            padding=True
        )
        initial_inputs = initial_inputs.to(self.edge_model.device).to(self.edge_model.model.dtype)
        
        # 当前上下文（会随着生成过程更新）
        current_inputs = {
            'input_ids': initial_inputs['input_ids'].clone(),
            'attention_mask': initial_inputs['attention_mask'].clone(),
        }
        if 'input_features' in initial_inputs:
            current_inputs['input_features'] = initial_inputs['input_features'].clone()
        if 'feature_attention_mask' in initial_inputs:
            current_inputs['feature_attention_mask'] = initial_inputs['feature_attention_mask'].clone()
        
        # 记录已接受的tokens
        all_accepted_tokens = []
        all_rejected_tokens = []
        
        edge_latency = 0.0
        cloud_latency = 0.0
        
        # Step 2: 逐块生成和验证
        for block_idx in range(max_blocks):
            logger.info(f"Processing block {block_idx + 1}/{max_blocks}")
            
            # 2.1 Edge模型生成候选tokens
            edge_start = time.time()
            candidate_tokens, candidate_log_probs = self._generate_candidate_tokens(
                current_inputs, block_size
            )
            edge_latency += time.time() - edge_start
            
            if not candidate_tokens:
                logger.info("No more tokens to generate, stopping")
                break
            
            # 2.2 计算不确定性，决定是否需要验证
            uncertainty = self._calculate_uncertainty(candidate_tokens, candidate_log_probs)
            needs_verification = self._should_verify_block(uncertainty, candidate_tokens)
            
            if needs_verification:
                logger.info(f"Block {block_idx} needs verification, entropy: {uncertainty.get('entropy', 0):.3f}")
                
                # 2.3 Cloud模型验证候选tokens（关键：传递完整上下文！）
                cloud_start = time.time()
                verification_result = self._verify_candidate_tokens_with_cloud(
                    current_inputs, candidate_tokens, audio_waveform, prompt
                )
                cloud_latency += time.time() - cloud_start
                
                # 2.4 接受/拒绝tokens
                accepted_tokens, rejected_tokens = self._accept_reject_tokens(
                    candidate_tokens, verification_result
                )
                
                logger.info(f"Block {block_idx}: {len(accepted_tokens)} accepted, {len(rejected_tokens)} rejected")
                
            else:
                logger.info(f"Block {block_idx} doesn't need verification, accepting all tokens")
                accepted_tokens = candidate_tokens
                rejected_tokens = []
            
            # 2.5 更新上下文（关键：将接受的tokens添加到上下文中）
            if accepted_tokens:
                self._update_context_with_accepted_tokens(current_inputs, accepted_tokens)
                all_accepted_tokens.extend(accepted_tokens)
            
            all_rejected_tokens.extend(rejected_tokens)
            
            # 检查是否遇到EOS token
            if any(token == self.edge_model.processor.tokenizer.eos_token_id for token in accepted_tokens):
                logger.info("Encountered EOS token, stopping generation")
                break
        
        # Step 3: 生成最终文本
        final_text = self.edge_model.processor.tokenizer.decode(
            all_accepted_tokens, skip_special_tokens=True
        )
        
        total_latency = time.time() - total_start_time
        total_tokens = len(all_accepted_tokens) + len(all_rejected_tokens)
        acceptance_rate = len(all_accepted_tokens) / max(total_tokens, 1)
        tokens_per_second = len(all_accepted_tokens) / max(total_latency, 0.001)
        
        logger.info(f"Correct speculative decoding completed:")
        logger.info(f"  Total latency: {total_latency:.3f}s")
        logger.info(f"  Edge latency: {edge_latency:.3f}s")
        logger.info(f"  Cloud latency: {cloud_latency:.3f}s")
        logger.info(f"  Acceptance rate: {acceptance_rate:.2%}")
        logger.info(f"  Tokens per second: {tokens_per_second:.2f}")
        
        return CorrectSpeculativeResult(
            final_text=final_text,
            accepted_tokens=all_accepted_tokens,
            rejected_tokens=all_rejected_tokens,
            total_latency=total_latency,
            edge_latency=edge_latency,
            cloud_latency=cloud_latency,
            acceptance_rate=acceptance_rate,
            tokens_per_second=tokens_per_second
        )
    
    def _generate_candidate_tokens(self, current_inputs: Dict, block_size: int) -> Tuple[List[int], List[float]]:
        """
        Edge模型生成候选tokens
        
        Args:
            current_inputs: 当前上下文输入
            block_size: 要生成的token数量
            
        Returns:
            (candidate_tokens, candidate_log_probs)
        """
        candidate_tokens = []
        candidate_log_probs = []
        
        # 复制当前输入
        generate_inputs = {k: v.clone() for k, v in current_inputs.items()}
        
        for _ in range(block_size):
            # 生成下一个token
            outputs = self.edge_model.model.generate(
                **generate_inputs,
                max_new_tokens=1,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.edge_model.processor.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
                return_audio=False
            )
            
            # 提取新token
            new_token = outputs.sequences[0][-1].item()
            
            # 检查EOS token
            if new_token == self.edge_model.processor.tokenizer.eos_token_id:
                break
            
            # 获取log概率
            if outputs.scores:
                logits = outputs.scores[0][0]
                log_probs = torch.log_softmax(logits, dim=-1)
                token_log_prob = log_probs[new_token].item()
            else:
                token_log_prob = 0.0
            
            candidate_tokens.append(new_token)
            candidate_log_probs.append(token_log_prob)
            
            # 更新输入用于下一个token
            new_token_tensor = torch.tensor([[new_token]], device=self.edge_model.device)
            generate_inputs['input_ids'] = torch.cat([generate_inputs['input_ids'], new_token_tensor], dim=1)
            generate_inputs['attention_mask'] = torch.cat([generate_inputs['attention_mask'], torch.ones((1, 1), device=self.edge_model.device)], dim=1)
        
        return candidate_tokens, candidate_log_probs
    
    def _verify_candidate_tokens_with_cloud(self, 
                                          current_inputs: Dict,
                                          candidate_tokens: List[int],
                                          audio_waveform: torch.Tensor,
                                          prompt: str) -> Dict:
        """
        关键方法：Cloud模型验证候选tokens
        
        这里演示正确的上下文传递：
        1. Cloud模型接收完整的当前上下文
        2. Cloud模型知道Edge模型已经生成了哪些tokens
        3. Cloud模型验证这些候选tokens
        """
        logger.info(f"Cloud model verifying {len(candidate_tokens)} candidate tokens")
        
        # 关键：构建包含候选tokens的完整上下文
        # 1. 获取当前已生成的文本
        current_tokens = current_inputs['input_ids'][0].tolist()
        current_text = self.cloud_model.processor.tokenizer.decode(current_tokens, skip_special_tokens=True)
        
        # 2. 将候选tokens转换为文本
        candidate_text = self.cloud_model.processor.tokenizer.decode(candidate_tokens, skip_special_tokens=True)
        
        # 3. 构建包含候选tokens的完整prompt
        full_prompt = f"{prompt}\n\nCurrent response so far: {current_text}\n\nCandidate continuation: {candidate_text}\n\nPlease verify if this continuation is appropriate and provide the correct version if needed."
        
        # 4. Cloud模型基于完整上下文生成验证结果
        verification_text, cloud_metrics = self.cloud_model.generate_independently(
            audio_waveform=audio_waveform,
            prompt=full_prompt,
            max_new_tokens=len(candidate_tokens) + 2,  # 生成略多于候选tokens的数量
            temperature=0.7,
            top_p=0.9
        )
        
        # 5. 提取验证后的tokens
        verification_tokens = self.cloud_model.processor.tokenizer.encode(
            verification_text, add_special_tokens=False
        )
        
        return {
            'candidate_tokens': candidate_tokens,
            'verification_tokens': verification_tokens,
            'verification_text': verification_text,
            'cloud_metrics': cloud_metrics
        }
    
    def _accept_reject_tokens(self, candidate_tokens: List[int], verification_result: Dict) -> Tuple[List[int], List[int]]:
        """
        接受/拒绝tokens的机制
        """
        verification_tokens = verification_result['verification_tokens']
        
        accepted_tokens = []
        rejected_tokens = []
        
        # 简单的token-by-token比较
        min_length = min(len(candidate_tokens), len(verification_tokens))
        
        for i in range(min_length):
            if candidate_tokens[i] == verification_tokens[i]:
                accepted_tokens.append(candidate_tokens[i])
            else:
                rejected_tokens.append(candidate_tokens[i])
                accepted_tokens.append(verification_tokens[i])
        
        # 处理剩余tokens
        if len(candidate_tokens) > min_length:
            rejected_tokens.extend(candidate_tokens[min_length:])
        elif len(verification_tokens) > min_length:
            accepted_tokens.extend(verification_tokens[min_length:])
        
        return accepted_tokens, rejected_tokens
    
    def _update_context_with_accepted_tokens(self, current_inputs: Dict, accepted_tokens: List[int]):
        """
        关键方法：将接受的tokens添加到上下文中
        
        这是speculative decoding的核心：上下文必须随着接受的tokens更新
        """
        if not accepted_tokens:
            return
        
        # 将接受的tokens添加到input_ids
        new_tokens_tensor = torch.tensor([accepted_tokens], device=self.edge_model.device)
        current_inputs['input_ids'] = torch.cat([current_inputs['input_ids'], new_tokens_tensor], dim=1)
        
        # 更新attention_mask
        new_attention = torch.ones((1, len(accepted_tokens)), device=self.edge_model.device)
        current_inputs['attention_mask'] = torch.cat([current_inputs['attention_mask'], new_attention], dim=1)
        
        logger.info(f"Updated context with {len(accepted_tokens)} accepted tokens")
    
    def _calculate_uncertainty(self, tokens: List[int], log_probs: List[float]) -> Dict:
        """计算不确定性指标"""
        if not log_probs:
            return {'entropy': 0.0, 'margin': 0.0}
        
        # 计算熵
        probs = [torch.exp(torch.tensor(lp)) for lp in log_probs]
        entropy = -sum(p * torch.log(p + 1e-8) for p in probs).item()
        
        # 计算边际（简化版本）
        margin = min(log_probs) if log_probs else 0.0
        
        return {
            'entropy': entropy,
            'margin': margin,
            'token_log_probs': log_probs
        }
    
    def _should_verify_block(self, uncertainty: Dict, tokens: List[int]) -> bool:
        """决定是否需要验证"""
        entropy = uncertainty.get('entropy', 0)
        return entropy > 1.0  # 简单的阈值

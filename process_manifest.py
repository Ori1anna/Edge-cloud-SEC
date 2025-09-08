import json
import re

def process_caption(caption):
    """
    处理caption，只保留音频相关的描述
    """
    if not caption:
        return caption
    
    # 分割caption为不同部分
    parts = caption.split('；')
    
    # 只保留包含"在音频中"的部分
    audio_parts = []
    for part in parts:
        part = part.strip()
        if part.startswith('在音频中'):
            audio_parts.append(part)
    
    # 处理第二部分：删除与视频相关的描述
    # 查找包含"根据视频线索"、"视频线索"等关键词的部分并删除
    processed_parts = []
    for part in audio_parts:
        # 删除包含视频相关关键词的句子
        sentences = part.split('。')
        filtered_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                # 检查是否包含视频相关的关键词
                video_keywords = [
                    '根据视频线索', '视频线索', '视频中', '画面', '面部表情', 
                    '眼神', '眉头', '嘴角', '嘴巴', '头部', '身体', '手势',
                    '动作', '姿态', '环境', '背景', '光线', '场景', '镜头',
                    '特写', '画面显示', '画面中', '视频开头', '视频中间',
                    '视频结尾', '视频最后', '视频的', '视频应该', '视频是',
                    '视频出现', '视频进行', '视频变化', '视频内容'
                ]
                
                # 如果句子包含视频相关关键词，跳过
                has_video_keyword = any(keyword in sentence for keyword in video_keywords)
                if not has_video_keyword:
                    filtered_sentences.append(sentence)
        
        # 重新组合句子
        if filtered_sentences:
            processed_part = '。'.join(filtered_sentences) + '。'
            processed_parts.append(processed_part)
    
    # 重新组合所有部分
    if processed_parts:
        return '；'.join(processed_parts) + '；'
    else:
        return ""

def process_english_caption(caption):
    """
    处理英文caption，只保留音频相关的描述
    """
    if not caption:
        return caption
    
    # 分割caption为不同部分
    parts = caption.split('. ')
    
    # 只保留包含"In the audio"的部分
    audio_parts = []
    for part in parts:
        part = part.strip()
        if part.startswith('In the audio'):
            audio_parts.append(part)
    
    # 处理第二部分：删除与视频相关的描述
    processed_parts = []
    for part in audio_parts:
        # 删除包含视频相关关键词的句子
        sentences = part.split('. ')
        filtered_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                # 检查是否包含视频相关的关键词
                video_keywords = [
                    'In the video', 'video', 'scene', 'screen', 'facial expression',
                    'eyes', 'eyebrows', 'mouth', 'head', 'body', 'gesture',
                    'action', 'posture', 'environment', 'background', 'lighting',
                    'camera', 'shot', 'close-up', 'scene shows', 'opening scene',
                    'middle part', 'end of the video', 'towards the end',
                    'as the video', 'video appears', 'video takes place',
                    'video progresses', 'video changes', 'video content'
                ]
                
                # 如果句子包含视频相关关键词，跳过
                has_video_keyword = any(keyword.lower() in sentence.lower() for keyword in video_keywords)
                if not has_video_keyword:
                    filtered_sentences.append(sentence)
        
        # 重新组合句子
        if filtered_sentences:
            processed_part = '. '.join(filtered_sentences)
            processed_parts.append(processed_part)
    
    # 重新组合所有部分
    if processed_parts:
        return '. '.join(processed_parts) + '.'
    else:
        return ""

def process_manifest_file(input_file, output_file):
    """
    处理manifest.json文件
    """
    print(f"正在处理文件: {input_file}")
    
    # 读取JSON文件
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"总共找到 {len(data)} 条记录")
    
    # 处理每条记录
    processed_count = 0
    for i, item in enumerate(data):
        if i % 100 == 0:
            print(f"已处理 {i} 条记录...")
        
        # 处理中文caption
        if 'chinese_caption' in item:
            original_chinese = item['chinese_caption']
            item['chinese_caption'] = process_caption(original_chinese)
            if original_chinese != item['chinese_caption']:
                processed_count += 1
        
        # 处理英文caption
        if 'english_caption' in item:
            original_english = item['english_caption']
            item['english_caption'] = process_english_caption(original_english)
            if original_english != item['english_caption']:
                processed_count += 1
    
    # 保存处理后的文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"处理完成！共处理了 {processed_count} 条记录")
    print(f"结果已保存到: {output_file}")

if __name__ == "__main__":
    input_file = "data/processed/mer2024/manifest.json"
    output_file = "data/processed/mer2024/manifest_audio_only.json"
    
    process_manifest_file(input_file, output_file)

















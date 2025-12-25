#!/usr/bin/env python3
"""
批量更新 exp_*.md 文件的 header 格式
统一为简洁的新格式
"""

import os
import re
from pathlib import Path
from datetime import datetime

LOGG_DIR = Path("/home/swei20/Physics_Informed_AI/logg")

# 新 header 模板
NEW_HEADER_TEMPLATE = '''# 📘 {title}
> **Name:** {name} | **ID:** `{exp_id}`  
> **Topic:** `{topic}` | **MVP:** {mvp} | **Project:** `{project}`  
> **Author:** {author} | **Date:** {date} | **Status:** {status}
```
💡 {purpose}  
决定：{decision}
```

---'''

def extract_info_from_old_header(content: str, filename: str) -> dict:
    """从旧格式提取信息"""
    info = {
        'title': 'Experiment Report',
        'name': 'TODO',
        'exp_id': '',
        'topic': '',
        'mvp': 'MVP-X.X',
        'project': 'VIT',
        'author': 'Viska Wei',
        'date': '',
        'status': '🔄',
        'purpose': '实验目的',
        'decision': '影响的决策'
    }
    
    # 从文件名提取 topic 和日期
    match = re.search(r'exp_([a-z_]+)_(\d{8})\.md', filename)
    if match:
        info['topic'] = match.group(1).split('_')[0]
        date_str = match.group(2)
        info['date'] = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
    
    # 提取标题
    title_match = re.search(r'^# 📘\s*(.+?)(?:\n|$)', content, re.MULTILINE)
    if title_match:
        info['title'] = title_match.group(1).strip()
    else:
        title_match = re.search(r'^#\s+(.+?)(?:\n|$)', content, re.MULTILINE)
        if title_match:
            info['title'] = title_match.group(1).strip()
    
    # 提取 Name
    name_match = re.search(r'\*\*Name:\*\*\s*(.+?)(?:\s*\||\s*$)', content)
    if name_match:
        info['name'] = name_match.group(1).strip()
    
    # 提取 ID
    id_match = re.search(r'\*\*ID:\*\*\s*`?([^`\n|]+)`?', content)
    if id_match:
        info['exp_id'] = id_match.group(1).strip()
    else:
        # 尝试生成 ID
        info['exp_id'] = f"VIT-{info['date'].replace('-', '')}-{info['topic']}-01" if info['date'] else 'TODO'
    
    # 提取 Topic
    topic_match = re.search(r'\*\*Topic[^:]*:\*\*\s*`?([^`\n|]+)`?', content)
    if topic_match:
        topic_val = topic_match.group(1).strip()
        if 'MVP' not in topic_val:
            info['topic'] = topic_val
    
    # 提取 MVP
    mvp_match = re.search(r'\*\*MVP:\*\*\s*([^\n|]+)', content)
    if mvp_match:
        info['mvp'] = mvp_match.group(1).strip()
    else:
        mvp_match = re.search(r'MVP[- ]?(\d+\.?\d*[A-Z]?)', content)
        if mvp_match:
            info['mvp'] = f"MVP-{mvp_match.group(1)}"
    
    # 提取 Author
    author_match = re.search(r'\*\*Author:\*\*\s*([^\n|]+)', content)
    if author_match:
        info['author'] = author_match.group(1).strip()
    
    # 提取 Date
    date_match = re.search(r'\*\*Date:\*\*\s*(\d{4}-\d{2}-\d{2})', content)
    if date_match:
        info['date'] = date_match.group(1)
    
    # 提取 Status
    status_match = re.search(r'\*\*Status:\*\*\s*([^\n|]+)', content)
    if status_match:
        info['status'] = status_match.group(1).strip()
    
    # 提取 Project
    project_match = re.search(r'\*\*Project:\*\*\s*`?([^`\n|]+)`?', content)
    if project_match:
        info['project'] = project_match.group(1).strip()
    
    # 提取一句话总结
    summary_match = re.search(r'##?\s*一句话[总結总结]?\s*\n+>\s*\*?\*?(.+?)(?:\n|$)', content)
    if summary_match:
        info['purpose'] = summary_match.group(1).strip()[:80]
    else:
        summary_match = re.search(r'\*\*一句话\*\*[：:]\s*(.+?)(?:\n|$)', content)
        if summary_match:
            info['purpose'] = summary_match.group(1).strip()[:80]
    
    return info

def find_header_end(content: str) -> int:
    """找到旧 header 结束的位置"""
    # 找到第一个实质性章节
    patterns = [
        r'^## 🔗\s*Upstream',
        r'^## ⚡\s*核心结论',
        r'^# ⚡\s*Key Findings',
        r'^#\s*1\.\s',
        r'^# 📑\s*Table of Contents',
        r'^---\s*\n\s*## 🔗',
        r'^---\s*\n\s*# ⚡',
    ]
    
    earliest = len(content)
    for pattern in patterns:
        match = re.search(pattern, content, re.MULTILINE)
        if match and match.start() < earliest:
            earliest = match.start()
    
    if earliest < len(content):
        return earliest
    
    # 如果找不到，尝试找第三个 ---
    dashes = list(re.finditer(r'^---\s*$', content, re.MULTILINE))
    if len(dashes) >= 2:
        return dashes[1].end()
    
    return 0

def update_exp_file(filepath: Path) -> bool:
    """更新单个 exp 文件"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查是否已经是新格式
        if re.search(r'^# 📘.*\n>\s*\*\*Name:\*\*.*\|.*\*\*ID:\*\*', content, re.MULTILINE):
            if '```\n💡' in content[:500]:
                print(f"  ⏭️  已是新格式: {filepath.name}")
                return False
        
        # 提取信息
        info = extract_info_from_old_header(content, filepath.name)
        
        # 生成新 header
        new_header = NEW_HEADER_TEMPLATE.format(**info)
        
        # 找到旧 header 结束位置
        header_end = find_header_end(content)
        
        if header_end > 0:
            # 保留 header 后的内容
            remaining = content[header_end:].lstrip('\n')
            # 确保有 Upstream Links 部分
            if '## 🔗 Upstream' not in remaining[:200]:
                remaining = '\n## 🔗 Upstream Links\n| Type | Link |\n|------|------|\n| 🧠 Hub | `logg/{topic}/{topic}_hub.md` |\n| 🗺️ Roadmap | `logg/{topic}/{topic}_roadmap.md` |\n\n---\n\n'.format(topic=info['topic']) + remaining
            new_content = new_header + '\n\n' + remaining
        else:
            # 无法找到结束点，只替换第一行
            first_newline = content.find('\n')
            new_content = new_header + '\n\n' + content[first_newline+1:]
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print(f"  ✅ 更新成功: {filepath.name}")
        return True
        
    except Exception as e:
        print(f"  ❌ 更新失败 {filepath.name}: {e}")
        return False

def main():
    print("🔄 开始批量更新 exp_*.md 文件 header...\n")
    
    # 收集所有 exp 文件
    exp_files = list(LOGG_DIR.rglob("exp_*.md"))
    print(f"📁 找到 {len(exp_files)} 个 exp 文件\n")
    
    updated = 0
    skipped = 0
    failed = 0
    
    for filepath in sorted(exp_files):
        result = update_exp_file(filepath)
        if result is True:
            updated += 1
        elif result is False:
            skipped += 1
        else:
            failed += 1
    
    print(f"\n📊 统计:")
    print(f"  ✅ 更新: {updated}")
    print(f"  ⏭️  跳过: {skipped}")
    print(f"  ❌ 失败: {failed}")

if __name__ == "__main__":
    main()


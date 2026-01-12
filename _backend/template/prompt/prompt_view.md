# 👁️ Markdown 预览（带语法高亮）

### 触发词
- `view` / `view <file_path>` / `预览`

### 用途
在浏览器中预览 Markdown 文件（GitHub 风格渲染 + Python 语法高亮）

### 服务器脚本
`_backend/scripts/md_server.py`

### 工作流程
1. 检查 md_server 是否运行（端口 8711）
   ```bash
   ps aux | grep "md_server.py.*8711" | grep -v grep
   ```
2. 如未运行，启动服务：
   ```bash
   cd /home/swei20/Physics_Informed_AI && python3 _backend/scripts/md_server.py 8711 &
   sleep 3
   ```
3. 返回预览 URL

### 输出格式
```
🌐 Markdown 预览服务运行中 (端口 8711)

📄 预览地址: http://localhost:8711/<relative_path>
```

### 管理命令
```bash
# 查看服务状态
ps aux | grep md_server

# 停止服务
pkill -f "md_server.py.*8711"
```

### 功能特性
- ✅ Python/JavaScript/Bash 等语法高亮 (One Dark 主题)
- ✅ 表格渲染
- ✅ 目录导航（可浏览文件夹）
- ✅ 响应式布局

### 备注
- SSH config 已配置 `LocalForward 8711 127.0.0.1:8711`，无需额外隧道
- 服务启动后会持续运行，多次 `view` 无需重启
- 代码高亮颜色：关键字紫色、函数蓝色、字符串绿色、数字橙色

# Stockholm ATC Simulator

🛩️ **先进的空中交通管制模拟器** - 基于斯德哥尔摩阿兰达机场(ESSA)终端区域

## 🎯 功能特性

### 核心功能
- ✈️ **实时航空器模拟** - 完整的飞行动力学模型
- 🗺️ **终端区域建模** - 精确的斯德哥尔摩TMA
- 📋 **SID/STAR程序** - 标准离场和进场程序
- 🎮 **实时控制** - WebSocket实时更新
- 📊 **统计监控** - 实时性能统计

### 技术特性
- 🌐 **Web界面** - 现代化的雷达显示界面
- 🖥️ **终端显示** - ASCII和图形化终端工具
- 🔄 **RESTful API** - 完整的程序化接口
- 📡 **WebSocket** - 实时数据推送
- 🎛️ **可配置** - 灵活的参数设置

## 🚀 快速开始

### 1. 安装依赖
```bash
pip install flask flask-socketio matplotlib numpy requests
```

### 2. 启动模拟器
```bash
# 完整模拟环境
python run_simulator.py --full

# 仅Web服务器
python run_simulator.py --web

# 仅终端显示
python run_simulator.py --terminal
```

### 3. 访问界面
- **Web界面**: http://localhost:8080
- **API**: http://localhost:8080/api/status

## 🏗️ 项目结构

```
atc_simulator/
├── core_simulator.py      # 核心模拟引擎
├── web_server.py          # Flask Web服务器
├── terminal_display.py    # 终端显示工具
├── run_simulator.py       # 启动脚本
├── templates/
│   └── index.html         # Web界面
└── README.md             # 说明文档
```

## 🌐 Web界面功能

### 主要组件
1. **交互式地图** - 基于Leaflet的实时雷达显示
2. **航空器管理** - 添加、移除、监控航空器
3. **程序管理** - SID/STAR程序分配
4. **统计面板** - 实时统计信息
5. **控制面板** - 模拟控制

### 视觉元素
- 🛩️ **机场标记** - ESSA阿兰达机场
- ✈️ **航空器图标** - 颜色编码的飞机位置
- 📍 **航路点** - SID/STAR航路点
- 🔵 **控制区域** - CTR和TMA边界

## 📺 终端显示

### 图形模式
```bash
python terminal_display.py
```
- 实时matplotlib图表
- 航空器轨迹显示
- 统计图表

### 文本模式
```bash
python terminal_display.py --text
```
- ASCII艺术地图
- 表格化航空器信息
- 实时统计更新

## 🛩️ 航空器和程序

### 支持的机型
- A320, A321, A333 (空客)
- B737, B738, B777 (波音)
- 其他商用机型

### SID程序 (标准离场)
```
ARS1A:    ES001 → SOLNA → ARS
ARS1B:    ES002 → SOLNA → ARS
HAPZI1A:  ES001 → VIKBY → HAPZI
HAPZI1B:  ES002 → VIKBY → HAPZI
ABENI1A:  ES003 → NOPEN → ABENI
ELTOK1A:  ES004 → RONVI → ELTOK
```

### STAR程序 (标准进场)
```
ARS1A:    ARS → SOLNA → ES001
ARS1B:    ARS → SOLNA → ES002
HAPZI1A:  HAPZI → VIKBY → ES002
ELTOK1A:  ELTOK → RONVI → ES004
ABENI1A:  ABENI → RIBSO → ES003
```

### 主要航路点
| 航路点 | 纬度 | 经度 | 类型 |
|--------|------|------|------|
| ELTOK | 59.4° | 17.5° | 进入点 |
| HAPZI | 59.7° | 18.2° | 进入点 |
| ARS | 59.8° | 17.8° | 航路点 |
| ABENI | 59.5° | 17.7° | 航路点 |
| RIBSO | 59.3° | 18.1° | 航路点 |

## 🔌 API 端点

### 模拟控制
- `GET /api/status` - 系统状态
- `GET /api/simulation/state` - 模拟状态
- `POST /api/simulation/start` - 启动模拟
- `POST /api/simulation/stop` - 停止模拟

### 航空器管理
- `POST /api/aircraft/add` - 添加航空器
- `POST /api/aircraft/remove` - 移除航空器

### 程序查询
- `GET /api/procedures/sids` - 获取SID程序
- `GET /api/procedures/stars` - 获取STAR程序

## 📊 使用示例

### 添加航空器 (API)
```bash
curl -X POST http://localhost:8080/api/aircraft/add \\
  -H "Content-Type: application/json" \\
  -d '{
    "callsign": "SAS123",
    "aircraft_type": "A320",
    "lat": 59.651,
    "lon": 17.918,
    "altitude": 1000,
    "heading": 13,
    "speed": 180,
    "sid": "ARS1A",
    "runway": "01L"
  }'
```

### WebSocket连接 (JavaScript)
```javascript
const socket = io('http://localhost:8080');

socket.on('simulation_update', function(data) {
    console.log('Aircraft count:', Object.keys(data.aircraft).length);
    console.log('Statistics:', data.stats);
});

socket.emit('add_aircraft', {
    callsign: 'NAX456',
    aircraft_type: 'B737',
    lat: 59.8,
    lon: 17.8,
    altitude: 15000,
    heading: 180,
    speed: 280,
    star: 'ARS1A'
});
```

## 🎮 操作指南

### Web界面操作
1. **启动模拟** - 点击"Start"按钮
2. **添加航空器** - 填写表单后点击"Add Aircraft"
3. **查看详情** - 点击地图上的航空器标记
4. **监控统计** - 查看左侧统计面板

### 示例场景
1. **离场场景** - 从ESSA起飞，分配SID
2. **进场场景** - 进入TMA，分配STAR
3. **过境场景** - 高空穿越TMA

## 🔧 配置选项

### 模拟参数
- 更新频率: 0.5秒
- 地图范围: 59.2°-60.1°N, 17.2°-18.7°E
- 高度范围: 地面 - FL410
- 速度范围: 100-500节

### 网络设置
- 默认端口: 8080
- WebSocket支持: 是
- CORS: 允许所有源

## 🐛 故障排除

### 常见问题
1. **端口占用** - 修改web_server.py中的端口设置
2. **无法连接** - 检查防火墙设置
3. **显示问题** - 安装matplotlib和相关依赖

### 调试模式
```bash
# 启用调试日志
export FLASK_DEBUG=1
python web_server.py
```

## 🚧 开发计划

### 即将推出
- [ ] 高级天气模拟
- [ ] 冲突检测和解脱
- [ ] 更多机场支持
- [ ] 历史数据回放
- [ ] 性能优化

### 长期计划
- [ ] 机器学习优化
- [ ] 多用户协作
- [ ] 移动端支持
- [ ] 云端部署

## 📄 许可证

MIT License - 自由使用和修改

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📞 支持

如有问题，请创建GitHub Issue或联系开发团队。

---

**🛩️ Happy Flying! 🛩️**
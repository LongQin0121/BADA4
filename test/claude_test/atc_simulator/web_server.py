#!/usr/bin/env python3
"""
Stockholm ATC Simulator - Web Server
基于Flask的Web界面服务器
端口: 8080
"""

from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import threading
import time
import json
from core_simulator import ATCSimulator
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'stockholm_atc_simulator'
socketio = SocketIO(app, cors_allowed_origins="*")

# 全局模拟器实例
simulator = ATCSimulator()
update_thread = None

def simulation_update_thread():
    """模拟更新线程"""
    while simulator.is_running:
        simulator.update_simulation()
        
        # 向所有客户端发送更新
        state = simulator.get_simulation_state()
        socketio.emit('simulation_update', state)
        
        time.sleep(0.5)  # 每0.5秒更新一次

@app.route('/')
def index():
    """主页"""
    return render_template('index.html')

@app.route('/api/status')
def api_status():
    """API状态"""
    return jsonify({
        'status': 'running',
        'aircraft_count': len(simulator.aircraft),
        'simulation_running': simulator.is_running
    })

@app.route('/api/simulation/state')
def api_simulation_state():
    """获取模拟状态"""
    return jsonify(simulator.get_simulation_state())

@app.route('/api/aircraft/add', methods=['POST'])
def api_add_aircraft():
    """添加航空器"""
    data = request.json
    try:
        aircraft = simulator.add_aircraft(
            data['callsign'],
            data['aircraft_type'],
            float(data['lat']),
            float(data['lon']),
            int(data['altitude']),
            float(data['heading']),
            int(data['speed'])
        )
        
        # 如果指定了SID或STAR，分配程序
        if 'sid' in data and data['sid']:
            simulator.assign_sid(data['callsign'], data['sid'], data.get('runway', '01L'))
        
        if 'star' in data and data['star']:
            simulator.assign_star(data['callsign'], data['star'], data.get('runway', '19R'))
            
        return jsonify({'success': True, 'message': f'Aircraft {data["callsign"]} added'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/aircraft/remove', methods=['POST'])
def api_remove_aircraft():
    """移除航空器"""
    data = request.json
    try:
        simulator.remove_aircraft(data['callsign'])
        return jsonify({'success': True, 'message': f'Aircraft {data["callsign"]} removed'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/simulation/start', methods=['POST'])
def api_start_simulation():
    """启动模拟"""
    global update_thread
    
    if not simulator.is_running:
        simulator.start_simulation()
        update_thread = threading.Thread(target=simulation_update_thread)
        update_thread.daemon = True
        update_thread.start()
        return jsonify({'success': True, 'message': 'Simulation started'})
    else:
        return jsonify({'success': False, 'error': 'Simulation already running'})

@app.route('/api/simulation/stop', methods=['POST'])
def api_stop_simulation():
    """停止模拟"""
    simulator.stop_simulation()
    return jsonify({'success': True, 'message': 'Simulation stopped'})

@app.route('/api/procedures/sids')
def api_get_sids():
    """获取SID程序"""
    return jsonify(simulator.tma.sids)

@app.route('/api/procedures/stars')
def api_get_stars():
    """获取STAR程序"""
    return jsonify(simulator.tma.stars)

@socketio.on('connect')
def handle_connect():
    """客户端连接"""
    logger.info('Client connected')
    emit('simulation_update', simulator.get_simulation_state())

@socketio.on('disconnect')
def handle_disconnect():
    """客户端断开"""
    logger.info('Client disconnected')

@socketio.on('add_aircraft')
def handle_add_aircraft(data):
    """处理添加航空器请求"""
    try:
        aircraft = simulator.add_aircraft(
            data['callsign'],
            data['aircraft_type'],
            float(data['lat']),
            float(data['lon']),
            int(data['altitude']),
            float(data['heading']),
            int(data['speed'])
        )
        
        if 'sid' in data and data['sid']:
            simulator.assign_sid(data['callsign'], data['sid'], data.get('runway', '01L'))
        
        if 'star' in data and data['star']:
            simulator.assign_star(data['callsign'], data['star'], data.get('runway', '19R'))
            
        emit('aircraft_added', {'success': True, 'callsign': data['callsign']})
        
    except Exception as e:
        emit('aircraft_added', {'success': False, 'error': str(e)})

def create_sample_scenarios():
    """创建示例场景"""
    # 清除现有航空器
    simulator.aircraft.clear()
    
    # 离场航空器
    aircraft1 = simulator.add_aircraft("SAS123", "A320", 59.651, 17.918, 1000, 13, 180)
    simulator.assign_sid("SAS123", "ARS1A", "01L")
    
    aircraft2 = simulator.add_aircraft("NAX456", "B737", 59.651, 17.918, 1500, 80, 200)
    simulator.assign_sid("NAX456", "HAPZI1A", "08")
    
    # 进场航空器
    aircraft3 = simulator.add_aircraft("SAS789", "A333", 59.8, 17.8, 15000, 180, 280)
    simulator.assign_star("SAS789", "ARS1A", "19R")
    
    aircraft4 = simulator.add_aircraft("LH321", "A320", 59.4, 17.5, 12000, 45, 250)
    simulator.assign_star("LH321", "ELTOK1A", "26")
    
    # 过境航空器
    aircraft5 = simulator.add_aircraft("KLM567", "B777", 59.7, 18.2, 35000, 270, 450)
    
    logger.info("Sample scenarios created")

if __name__ == '__main__':
    # 创建示例场景
    create_sample_scenarios()
    
    print("=== Stockholm ATC Simulator Web Server ===")
    print("Server starting on http://localhost:8080")
    print("Features:")
    print("- Real-time aircraft tracking")
    print("- SID/STAR procedures")
    print("- Interactive web interface")
    print("- WebSocket real-time updates")
    print("- RESTful API")
    print("\nPress Ctrl+C to stop")
    
    # 启动服务器
    socketio.run(app, host='0.0.0.0', port=8080, debug=False)
#!/usr/bin/env python3
"""
Stockholm ATC Simulator - 启动脚本
一键启动完整的ATC模拟环境
"""

import subprocess
import threading
import time
import sys
import os
import webbrowser
from core_simulator import ATCSimulator
import signal

class SimulatorLauncher:
    """模拟器启动器"""
    
    def __init__(self):
        self.processes = []
        self.running = True
        
    def signal_handler(self, signum, frame):
        """信号处理器"""
        print("\n🛑 Shutting down simulator...")
        self.running = False
        for process in self.processes:
            try:
                process.terminate()
            except:
                pass
        sys.exit(0)
    
    def start_web_server(self):
        """启动Web服务器"""
        print("🌐 Starting web server...")
        try:
            process = subprocess.Popen([sys.executable, 'web_server.py'], 
                                     cwd=os.path.dirname(__file__))
            self.processes.append(process)
            return True
        except Exception as e:
            print(f"❌ Failed to start web server: {e}")
            return False
    
    def start_terminal_display(self, mode='text'):
        """启动终端显示"""
        print(f"📺 Starting terminal display ({mode} mode)...")
        try:
            args = [sys.executable, 'terminal_display.py']
            if mode == 'text':
                args.append('--text')
            
            process = subprocess.Popen(args, cwd=os.path.dirname(__file__))
            self.processes.append(process)
            return True
        except Exception as e:
            print(f"❌ Failed to start terminal display: {e}")
            return False
    
    def wait_for_server(self, max_wait=10):
        """等待服务器启动"""
        print("⏳ Waiting for server to start...")
        import requests
        
        for i in range(max_wait):
            try:
                response = requests.get('http://localhost:8080/api/status', timeout=1)
                if response.status_code == 200:
                    print("✅ Server is ready!")
                    return True
            except:
                pass
            time.sleep(1)
            print(f"   Waiting... ({i+1}/{max_wait})")
        
        print("❌ Server failed to start")
        return False
    
    def open_browser(self):
        """打开浏览器"""
        try:
            webbrowser.open('http://localhost:8080')
            print("🌐 Browser opened at http://localhost:8080")
        except:
            print("📝 Please open your browser and go to: http://localhost:8080")
    
    def show_info(self):
        """显示信息"""
        print("=" * 60)
        print("🛩️  STOCKHOLM ATC SIMULATOR")
        print("=" * 60)
        print("🎯 Features:")
        print("   • Real-time aircraft simulation")
        print("   • SID/STAR procedures")
        print("   • Stockholm Arlanda (ESSA) terminal area")
        print("   • Web-based radar display")
        print("   • Terminal monitoring tools")
        print()
        print("🌐 Access Points:")
        print("   • Web Interface: http://localhost:8080")
        print("   • API Endpoint: http://localhost:8080/api/status")
        print()
        print("🎮 Controls:")
        print("   • Add aircraft via web interface")
        print("   • Monitor via terminal display")
        print("   • Real-time updates via WebSocket")
        print()
        print("🛑 Press Ctrl+C to stop all services")
        print("=" * 60)
    
    def run_full_simulation(self):
        """运行完整模拟"""
        # 设置信号处理
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        self.show_info()
        
        # 启动Web服务器
        if not self.start_web_server():
            return
        
        # 等待服务器启动
        if not self.wait_for_server():
            return
        
        # 启动终端显示
        mode = input("\n📺 Terminal display mode [text/graph/none]: ").lower()
        if mode in ['text', 'graph']:
            self.start_terminal_display(mode if mode == 'text' else 'graph')
        
        # 询问是否打开浏览器
        open_browser = input("🌐 Open browser? [y/N]: ").lower()
        if open_browser in ['y', 'yes']:
            time.sleep(2)  # 等待服务器完全启动
            self.open_browser()
        
        print("\n✅ Simulator is running!")
        print("🌐 Web Interface: http://localhost:8080")
        print("🛑 Press Ctrl+C to stop")
        
        # 保持运行
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            self.signal_handler(signal.SIGINT, None)
    
    def run_web_only(self):
        """只运行Web服务器"""
        signal.signal(signal.SIGINT, self.signal_handler)
        
        print("🌐 Starting web server only...")
        if self.start_web_server():
            if self.wait_for_server():
                print("✅ Web server running at http://localhost:8080")
                print("🛑 Press Ctrl+C to stop")
                try:
                    while self.running:
                        time.sleep(1)
                except KeyboardInterrupt:
                    self.signal_handler(signal.SIGINT, None)
    
    def run_terminal_only(self):
        """只运行终端显示"""
        print("📺 Starting terminal display only...")
        print("Make sure web server is running at http://localhost:8080")
        
        mode = input("Display mode [text/graph]: ").lower()
        if mode not in ['text', 'graph']:
            mode = 'text'
        
        try:
            if mode == 'text':
                from terminal_display import TerminalTextDisplay
                display = TerminalTextDisplay()
                display.run()
            else:
                from terminal_display import TerminalDisplay
                display = TerminalDisplay()
                display.start_display()
        except ImportError as e:
            print(f"❌ Import error: {e}")
            print("Falling back to text mode...")
            from terminal_display import TerminalTextDisplay
            display = TerminalTextDisplay()
            display.run()

def show_usage():
    """显示使用说明"""
    print("Stockholm ATC Simulator - Launcher")
    print("Usage:")
    print("  python run_simulator.py [option]")
    print()
    print("Options:")
    print("  --full      Start complete simulation (web + terminal)")
    print("  --web       Start web server only")
    print("  --terminal  Start terminal display only")
    print("  --help      Show this help")
    print()
    print("Default (no option): Interactive mode")

def main():
    """主函数"""
    launcher = SimulatorLauncher()
    
    if len(sys.argv) > 1:
        option = sys.argv[1]
        
        if option == '--full':
            launcher.run_full_simulation()
        elif option == '--web':
            launcher.run_web_only()
        elif option == '--terminal':
            launcher.run_terminal_only()
        elif option == '--help':
            show_usage()
        else:
            print(f"Unknown option: {option}")
            show_usage()
    else:
        # 交互模式
        print("🛩️  Stockholm ATC Simulator")
        print("1. Full simulation (web + terminal)")
        print("2. Web server only")
        print("3. Terminal display only")
        
        choice = input("\nSelect option [1-3]: ")
        
        if choice == '1':
            launcher.run_full_simulation()
        elif choice == '2':
            launcher.run_web_only()
        elif choice == '3':
            launcher.run_terminal_only()
        else:
            print("Invalid choice")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Stockholm ATC Simulator - Core Engine
先进的空中交通管制模拟器核心
"""

import time
import threading
import json
import math
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Waypoint:
    """航路点类"""
    def __init__(self, name: str, lat: float, lon: float, waypoint_type: str = "waypoint"):
        self.name = name
        self.lat = lat
        self.lon = lon
        self.type = waypoint_type  # waypoint, vor, ndb, fix
        
    def distance_to(self, other_lat: float, other_lon: float) -> float:
        """计算到另一点的距离（海里）"""
        R = 3440.065  # 地球半径（海里）
        lat1_rad = math.radians(self.lat)
        lon1_rad = math.radians(self.lon)
        lat2_rad = math.radians(other_lat)
        lon2_rad = math.radians(other_lon)
        
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        return R * c
    
    def bearing_to(self, other_lat: float, other_lon: float) -> float:
        """计算到另一点的航向（度）"""
        lat1_rad = math.radians(self.lat)
        lon1_rad = math.radians(self.lon)
        lat2_rad = math.radians(other_lat)
        lon2_rad = math.radians(other_lon)
        
        dlon = lon2_rad - lon1_rad
        
        y = math.sin(dlon) * math.cos(lat2_rad)
        x = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon)
        
        bearing = math.atan2(y, x)
        bearing = math.degrees(bearing)
        bearing = (bearing + 360) % 360
        
        return bearing

class Aircraft:
    """航空器类"""
    def __init__(self, callsign: str, aircraft_type: str, lat: float, lon: float, 
                 altitude: int, heading: float, speed: int):
        self.callsign = callsign
        self.aircraft_type = aircraft_type
        self.lat = lat
        self.lon = lon
        self.altitude = altitude  # feet
        self.heading = heading    # degrees
        self.speed = speed        # knots
        
        # 飞行计划
        self.route: List[Waypoint] = []
        self.current_waypoint_index = 0
        self.target_altitude = altitude
        self.target_speed = speed
        
        # 状态
        self.is_climbing = False
        self.is_descending = False
        self.departure_airport = None
        self.arrival_airport = None
        self.sid = None
        self.star = None
        
        # 时间戳
        self.last_update = time.time()
        
    def add_waypoint(self, waypoint: Waypoint):
        """添加航路点"""
        self.route.append(waypoint)
        
    def get_current_waypoint(self) -> Optional[Waypoint]:
        """获取当前目标航路点"""
        if self.current_waypoint_index < len(self.route):
            return self.route[self.current_waypoint_index]
        return None
    
    def update_position(self, dt: float):
        """更新飞机位置"""
        # 计算新位置
        distance_nm = (self.speed * dt) / 3600  # 海里/小时 -> 海里/秒
        
        # 转换为纬度经度变化
        lat_change = (distance_nm * math.cos(math.radians(self.heading))) / 60
        lon_change = (distance_nm * math.sin(math.radians(self.heading))) / (60 * math.cos(math.radians(self.lat)))
        
        self.lat += lat_change
        self.lon += lon_change
        
        # 更新高度
        if self.is_climbing and self.altitude < self.target_altitude:
            climb_rate = 1500  # ft/min
            self.altitude += (climb_rate * dt) / 60
            if self.altitude >= self.target_altitude:
                self.altitude = self.target_altitude
                self.is_climbing = False
                
        elif self.is_descending and self.altitude > self.target_altitude:
            descent_rate = 1000  # ft/min
            self.altitude -= (descent_rate * dt) / 60
            if self.altitude <= self.target_altitude:
                self.altitude = self.target_altitude
                self.is_descending = False
        
        # 检查是否到达航路点
        current_wp = self.get_current_waypoint()
        if current_wp:
            distance_to_wp = current_wp.distance_to(self.lat, self.lon)
            if distance_to_wp < 1.0:  # 1海里内算到达
                self.current_waypoint_index += 1
                logger.info(f"{self.callsign} reached waypoint {current_wp.name}")
                
                # 更新航向到下一个航路点
                next_wp = self.get_current_waypoint()
                if next_wp:
                    self.heading = current_wp.bearing_to(next_wp.lat, next_wp.lon)
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            'callsign': self.callsign,
            'aircraft_type': self.aircraft_type,
            'lat': self.lat,
            'lon': self.lon,
            'altitude': self.altitude,
            'heading': self.heading,
            'speed': self.speed,
            'target_altitude': self.target_altitude,
            'current_waypoint': self.get_current_waypoint().name if self.get_current_waypoint() else None,
            'route': [wp.name for wp in self.route],
            'sid': self.sid,
            'star': self.star
        }

class StockholmTMA:
    """斯德哥尔摩终端区域"""
    def __init__(self):
        # ESSA机场信息
        self.airport_lat = 59.651111
        self.airport_lon = 17.918611
        self.airport_elevation = 135
        
        # 跑道信息
        self.runways = {
            '01L': {'lat': 59.64, 'lon': 17.91, 'heading': 13, 'length': 3301},
            '19R': {'lat': 59.66, 'lon': 17.93, 'heading': 193, 'length': 3301},
            '08': {'lat': 59.65, 'lon': 17.90, 'heading': 80, 'length': 2500},
            '26': {'lat': 59.66, 'lon': 17.94, 'heading': 260, 'length': 2500},
            '01R': {'lat': 59.64, 'lon': 17.93, 'heading': 13, 'length': 2500},
            '19L': {'lat': 59.66, 'lon': 17.95, 'heading': 193, 'length': 2500}
        }
        
        # 航路点定义
        self.waypoints = self._create_waypoints()
        
        # SID程序
        self.sids = self._create_sids()
        
        # STAR程序
        self.stars = self._create_stars()
        
    def _create_waypoints(self) -> Dict[str, Waypoint]:
        """创建航路点"""
        waypoints = {}
        
        # 主要航路点（基于真实ESSA程序）
        wp_data = [
            ('ELTOK', 59.4, 17.5, 'entry_point'),
            ('HAPZI', 59.7, 18.2, 'entry_point'),
            ('ARS', 59.8, 17.8, 'waypoint'),
            ('ABENI', 59.5, 17.7, 'waypoint'),
            ('RIBSO', 59.3, 18.1, 'waypoint'),
            ('RONVI', 59.9, 17.4, 'waypoint'),
            ('SOLNA', 59.6, 18.0, 'waypoint'),
            ('VIKBY', 59.8, 18.3, 'waypoint'),
            ('NOPEN', 59.2, 17.9, 'waypoint'),
            ('EVMAX', 60.0, 17.6, 'waypoint'),
            # 机场附近点
            ('ES001', 59.6, 17.8, 'fix'),
            ('ES002', 59.7, 17.9, 'fix'),
            ('ES003', 59.5, 17.9, 'fix'),
            ('ES004', 59.6, 18.1, 'fix')
        ]
        
        for name, lat, lon, wp_type in wp_data:
            waypoints[name] = Waypoint(name, lat, lon, wp_type)
            
        return waypoints
    
    def _create_sids(self) -> Dict[str, List[str]]:
        """创建标准离场程序"""
        return {
            'ARS1A': ['ES001', 'SOLNA', 'ARS'],
            'ARS1B': ['ES002', 'SOLNA', 'ARS'],
            'HAPZI1A': ['ES001', 'VIKBY', 'HAPZI'],
            'HAPZI1B': ['ES002', 'VIKBY', 'HAPZI'],
            'ABENI1A': ['ES003', 'NOPEN', 'ABENI'],
            'ELTOK1A': ['ES004', 'RONVI', 'ELTOK']
        }
    
    def _create_stars(self) -> Dict[str, List[str]]:
        """创建标准进场程序"""
        return {
            'ARS1A': ['ARS', 'SOLNA', 'ES001'],
            'ARS1B': ['ARS', 'SOLNA', 'ES002'],
            'HAPZI1A': ['HAPZI', 'VIKBY', 'ES002'],
            'ELTOK1A': ['ELTOK', 'RONVI', 'ES004'],
            'ABENI1A': ['ABENI', 'RIBSO', 'ES003']
        }
    
    def get_sid_route(self, sid_name: str, runway: str) -> List[Waypoint]:
        """获取SID航路"""
        if sid_name not in self.sids:
            return []
        
        route = []
        for wp_name in self.sids[sid_name]:
            if wp_name in self.waypoints:
                route.append(self.waypoints[wp_name])
        
        return route
    
    def get_star_route(self, star_name: str, runway: str) -> List[Waypoint]:
        """获取STAR航路"""
        if star_name not in self.stars:
            return []
        
        route = []
        for wp_name in self.stars[star_name]:
            if wp_name in self.waypoints:
                route.append(self.waypoints[wp_name])
        
        return route

class ATCSimulator:
    """ATC模拟器主类"""
    def __init__(self):
        self.tma = StockholmTMA()
        self.aircraft: Dict[str, Aircraft] = {}
        self.is_running = False
        self.simulation_speed = 1.0  # 1x实时
        self.start_time = datetime.now()
        
        # 统计
        self.stats = {
            'total_aircraft': 0,
            'active_aircraft': 0,
            'departed': 0,
            'arrived': 0
        }
        
    def add_aircraft(self, callsign: str, aircraft_type: str, lat: float, lon: float,
                    altitude: int, heading: float, speed: int) -> Aircraft:
        """添加航空器"""
        aircraft = Aircraft(callsign, aircraft_type, lat, lon, altitude, heading, speed)
        self.aircraft[callsign] = aircraft
        self.stats['total_aircraft'] += 1
        self.stats['active_aircraft'] += 1
        logger.info(f"Added aircraft {callsign} at {lat:.4f}, {lon:.4f}")
        return aircraft
    
    def remove_aircraft(self, callsign: str):
        """移除航空器"""
        if callsign in self.aircraft:
            del self.aircraft[callsign]
            self.stats['active_aircraft'] -= 1
            logger.info(f"Removed aircraft {callsign}")
    
    def assign_sid(self, callsign: str, sid_name: str, runway: str):
        """分配SID"""
        if callsign in self.aircraft:
            aircraft = self.aircraft[callsign]
            aircraft.sid = sid_name
            route = self.tma.get_sid_route(sid_name, runway)
            aircraft.route.extend(route)
            logger.info(f"Assigned SID {sid_name} to {callsign}")
    
    def assign_star(self, callsign: str, star_name: str, runway: str):
        """分配STAR"""
        if callsign in self.aircraft:
            aircraft = self.aircraft[callsign]
            aircraft.star = star_name
            route = self.tma.get_star_route(star_name, runway)
            aircraft.route.extend(route)
            logger.info(f"Assigned STAR {star_name} to {callsign}")
    
    def update_simulation(self):
        """更新模拟"""
        current_time = time.time()
        
        for aircraft in list(self.aircraft.values()):
            dt = (current_time - aircraft.last_update) * self.simulation_speed
            aircraft.update_position(dt)
            aircraft.last_update = current_time
            
            # 检查是否需要移除航空器（离开区域）
            distance_from_center = math.sqrt(
                (aircraft.lat - self.tma.airport_lat)**2 + 
                (aircraft.lon - self.tma.airport_lon)**2
            )
            
            if distance_from_center > 1.0:  # 离开1度范围
                if len(aircraft.route) == 0 or aircraft.current_waypoint_index >= len(aircraft.route):
                    self.remove_aircraft(aircraft.callsign)
    
    def start_simulation(self):
        """启动模拟"""
        self.is_running = True
        logger.info("Simulation started")
    
    def stop_simulation(self):
        """停止模拟"""
        self.is_running = False
        logger.info("Simulation stopped")
    
    def get_simulation_state(self) -> Dict:
        """获取模拟状态"""
        return {
            'aircraft': {callsign: aircraft.to_dict() for callsign, aircraft in self.aircraft.items()},
            'waypoints': {name: {'lat': wp.lat, 'lon': wp.lon, 'type': wp.type} 
                         for name, wp in self.tma.waypoints.items()},
            'runways': self.tma.runways,
            'airport': {
                'lat': self.tma.airport_lat,
                'lon': self.tma.airport_lon,
                'elevation': self.tma.airport_elevation
            },
            'stats': self.stats,
            'simulation_time': str(datetime.now() - self.start_time),
            'is_running': self.is_running
        }

if __name__ == "__main__":
    # 测试代码
    sim = ATCSimulator()
    
    # 添加测试航空器
    aircraft1 = sim.add_aircraft("SAS123", "A320", 59.5, 17.6, 5000, 45, 250)
    sim.assign_sid("SAS123", "ARS1A", "01L")
    
    aircraft2 = sim.add_aircraft("NAX456", "B737", 59.8, 17.9, 15000, 180, 280)
    sim.assign_star("NAX456", "HAPZI1A", "19R")
    
    print("=== Stockholm ATC Simulator ===")
    print(f"Aircraft count: {len(sim.aircraft)}")
    print(f"Waypoints: {len(sim.tma.waypoints)}")
    print(f"SIDs: {len(sim.tma.sids)}")
    print(f"STARs: {len(sim.tma.stars)}")
    
    # 运行几次更新
    sim.start_simulation()
    for i in range(5):
        sim.update_simulation()
        time.sleep(0.1)
    
    state = sim.get_simulation_state()
    print(f"\nSimulation state: {json.dumps(state, indent=2)}")
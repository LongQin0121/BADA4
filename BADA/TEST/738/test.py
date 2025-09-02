#!/usr/bin/env python3
"""
完整可用的pyBADA Boeing 737-800参数获取方案
基于成功测试的方法2和方法3
"""

print("🛩️ Boeing 737-800 完整参数获取")
print("="*60)

from pyBADA.bada4 import BADA4, Airplane
import xml.etree.ElementTree as ET
from pathlib import Path
import pandas as pd

# 配置
BADA4_PATH = "/home/longqin/anaconda3/envs/atc_sim/lib/python3.12/site-packages/pyBADA/aircraft/BADA4"
AIRCRAFT_CODE = 'B738W26'  # Boeing 737-800

class AircraftParameterExtractor:
    """飞机参数提取器 - 整合两种成功方法"""
    
    def __init__(self, aircraft_code):
        self.aircraft_code = aircraft_code
        self.bada4_path = BADA4_PATH
        self.xml_path = f"{self.bada4_path}/Models/{aircraft_code}/{aircraft_code}.xml"
        
        # 初始化BADA对象
        self.airplane = None
        self.bada4 = None
        self.xml_params = None
        
    def initialize_bada4(self):
        """初始化BADA4对象（方法2）"""
        try:
            print(f"🔧 初始化BADA4对象...")
            
            # 创建Airplane实例
            self.airplane = Airplane()
            print("✅ Airplane创建成功")
            
            # 设置飞机属性
            self.airplane.aircraft_type = self.aircraft_code
            self.airplane.code = self.aircraft_code
            
            # 创建BADA4实例
            self.bada4 = BADA4(self.airplane)
            print("✅ BADA4创建成功")
            
            # 探索可用方法
            bada4_methods = [method for method in dir(self.bada4) 
                           if not method.startswith('_') and callable(getattr(self.bada4, method))]
            print(f"📋 BADA4可用方法: {len(bada4_methods)}个")
            print(f"   主要方法: {bada4_methods[:15]}")
            
            return True
            
        except Exception as e:
            print(f"❌ BADA4初始化失败: {e}")
            return False
    
    def extract_xml_parameters(self):
        """提取XML参数（方法3）"""
        try:
            print(f"\n📖 解析XML文件...")
            
            if not Path(self.xml_path).exists():
                print(f"❌ XML文件不存在: {self.xml_path}")
                return False
            
            # 解析XML
            tree = ET.parse(self.xml_path)
            root = tree.getroot()
            print(f"✅ XML解析成功: {root.tag}")
            
            # 提取所有参数
            self.xml_params = {}
            
            def extract_from_element(element, prefix=""):
                """递归提取XML元素的参数"""
                # 处理元素属性
                for attr_name, attr_value in element.attrib.items():
                    key = f"{prefix}{attr_name}" if prefix else attr_name
                    try:
                        self.xml_params[key] = float(attr_value)
                    except ValueError:
                        self.xml_params[key] = attr_value
                
                # 处理元素文本内容
                if element.text and element.text.strip():
                    text = element.text.strip()
                    if len(text) < 100:  # 避免太长的文本
                        key = f"{prefix}{element.tag.split('}')[-1]}" if prefix else element.tag.split('}')[-1]
                        try:
                            self.xml_params[key] = float(text)
                        except ValueError:
                            self.xml_params[key] = text
                
                # 递归处理子元素
                for child in element:
                    child_tag = child.tag.split('}')[-1]  # 去除命名空间
                    new_prefix = f"{prefix}{child_tag}_" if prefix else f"{child_tag}_"
                    extract_from_element(child, new_prefix)
            
            # 开始提取
            extract_from_element(root)
            
            print(f"📊 提取到 {len(self.xml_params)} 个XML参数")
            return True
            
        except Exception as e:
            print(f"❌ XML解析失败: {e}")
            return False
    
    def get_bada4_parameters(self):
        """从BADA4对象获取参数"""
        if not self.bada4:
            print("❌ BADA4对象未初始化")
            return {}
        
        print(f"\n🔧 从BADA4对象提取参数...")
        bada4_params = {}
        
        # 尝试获取各种参数
        parameter_methods = [
            ('CD', 'drag_coefficient'),
            ('CL', 'lift_coefficient'), 
            ('CF', 'fuel_consumption'),
            ('esf', 'engine_scale_factor')
        ]
        
        for method_name, param_name in parameter_methods:
            try:
                if hasattr(self.bada4, method_name):
                    method = getattr(self.bada4, method_name)
                    if callable(method):
                        # 尝试无参数调用
                        try:
                            result = method()
                            bada4_params[param_name] = result
                            print(f"✅ {param_name}: {result}")
                        except Exception as e:
                            print(f"⚠️ {param_name}需要参数: {e}")
                    else:
                        # 直接属性
                        bada4_params[param_name] = method
                        print(f"✅ {param_name}: {method}")
            except Exception as e:
                print(f"❌ 获取{param_name}失败: {e}")
        
        return bada4_params
    
    def get_complete_parameters(self):
        """获取完整的飞机参数集合"""
        print(f"\n🎯 获取{self.aircraft_code}完整参数...")
        
        # 初始化
        bada4_success = self.initialize_bada4()
        xml_success = self.extract_xml_parameters()
        
        if not (bada4_success or xml_success):
            print("❌ 所有方法都失败了")
            return None
        
        # 合并参数
        complete_params = {
            'aircraft_info': {
                'code': self.aircraft_code,
                'name': 'Boeing 737-800',
                'type': 'Commercial Aircraft'
            }
        }
        
        # 添加BADA4参数
        if bada4_success:
            bada4_params = self.get_bada4_parameters()
            if bada4_params:
                complete_params['bada4_parameters'] = bada4_params
        
        # 添加XML参数
        if xml_success and self.xml_params:
            # 按类别整理XML参数
            categorized_params = self.categorize_xml_params()
            complete_params.update(categorized_params)
        
        return complete_params
    
    def categorize_xml_params(self):
        """将XML参数按类别整理"""
        if not self.xml_params:
            return {}
        
        categories = {
            'geometry': {},
            'mass': {},
            'performance': {},
            'aerodynamics': {},
            'engine': {},
            'other': {}
        }
        
        # 参数分类规则
        category_keywords = {
            'geometry': ['length', 'wingspan', 'span', 'height', 'width', 'area', 'S'],
            'mass': ['mass', 'weight', 'MTOW', 'MLW', 'MZFW'],
            'performance': ['speed', 'velocity', 'altitude', 'ceiling', 'range', 'v', 'h', 'M'],
            'aerodynamics': ['CD', 'CL', 'drag', 'lift', 'alpha'],
            'engine': ['thrust', 'CF', 'fuel', 'power', 'N1', 'N2']
        }
        
        # 分类参数
        for param_name, param_value in self.xml_params.items():
            categorized = False
            param_lower = param_name.lower()
            
            for category, keywords in category_keywords.items():
                if any(keyword.lower() in param_lower for keyword in keywords):
                    categories[category][param_name] = param_value
                    categorized = True
                    break
            
            if not categorized:
                categories['other'][param_name] = param_value
        
        # 只返回非空类别
        return {cat: params for cat, params in categories.items() if params}

def main():
    """主函数 - 获取Boeing 737-800参数"""
    
    # 创建参数提取器
    extractor = AircraftParameterExtractor(AIRCRAFT_CODE)
    
    # 获取完整参数
    params = extractor.get_complete_parameters()
    
    if params:
        print(f"\n🎉 成功获取{AIRCRAFT_CODE}参数!")
        print("="*60)
        
        # 显示参数摘要
        for category, category_params in params.items():
            print(f"\n📊 {category.upper()}:")
            
            if isinstance(category_params, dict):
                for param_name, param_value in list(category_params.items())[:10]:  # 只显示前10个
                    print(f"   {param_name}: {param_value}")
                
                if len(category_params) > 10:
                    print(f"   ... 还有 {len(category_params) - 10} 个参数")
            else:
                print(f"   {category_params}")
        
        # 生成参数总结
        print(f"\n📋 参数总结:")
        total_params = sum(len(v) if isinstance(v, dict) else 1 for v in params.values())
        print(f"   总参数数: {total_params}")
        print(f"   参数类别: {len(params)}")
        
        # 保存到文件
        try:
            import json
            output_file = f"{AIRCRAFT_CODE}_parameters.json"
            
            # 转换numpy类型为Python基础类型
            def convert_types(obj):
                if hasattr(obj, 'item'):  # numpy types
                    return obj.item()
                elif isinstance(obj, dict):
                    return {k: convert_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_types(v) for v in obj]
                else:
                    return obj
            
            clean_params = convert_types(params)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(clean_params, f, indent=2, ensure_ascii=False)
            
            print(f"✅ 参数已保存到: {output_file}")
            
        except Exception as e:
            print(f"⚠️ 保存文件失败: {e}")
        
        return params
    
    else:
        print("❌ 未能获取参数")
        return None

def get_specific_parameters(aircraft_code, param_list):
    """获取特定参数的便捷函数"""
    extractor = AircraftParameterExtractor(aircraft_code)
    all_params = extractor.get_complete_parameters()
    
    if not all_params:
        return None
    
    # 搜索指定参数
    found_params = {}
    
    def search_in_dict(d, target_keys):
        results = {}
        for key, value in d.items():
            if isinstance(value, dict):
                nested_results = search_in_dict(value, target_keys)
                results.update(nested_results)
            else:
                for target_key in target_keys:
                    if target_key.lower() in key.lower():
                        results[key] = value
        return results
    
    found_params = search_in_dict(all_params, param_list)
    return found_params

# 运行主程序
if __name__ == "__main__":
    # 获取完整参数
    boeing_params = main()
    
    # 示例：获取特定参数
    print(f"\n🔍 获取特定参数示例:")
    specific_params = get_specific_parameters(AIRCRAFT_CODE, ['length', 'mass', 'speed', 'altitude'])
    
    if specific_params:
        print("找到的特定参数:")
        for param_name, param_value in specific_params.items():
            print(f"   {param_name}: {param_value}")
    
    print(f"\n🎯 使用建议:")
    print("1. 查看生成的JSON文件获取完整参数列表")
    print("2. 使用get_specific_parameters()函数获取你需要的特定参数")
    print("3. 修改AIRCRAFT_CODE变量测试其他飞机型号")
    print("4. 集成到你的ATC仿真系统中")
from pyBADA.bada4 import Bada4Aircraft

AC = Bada4Aircraft(
    badaVersion="4.2",
    acName="A320-232",
    filePath="/home/longqin/Downloads/4.2/BADA_4.2_L06514UPC/Models",
)

print("Aircraft loaded successfully!")
print(f"Aircraft type: {AC.acName}")
print(f"BADA version: {AC.BADAVersion}")
print(f"BADA family: {AC.BADAFamily}")
print(f"BADA family name: {AC.BADAFamilyName}")

# from pyBADA.bada4 import Bada4Aircraft

# AC = Bada4Aircraft(
#     badaVersion="4.2",
#     acName="A320-232",
#     filePath="/home/longqin/Downloads/4.2/BADA_4.2_L06514UPC/Models",
# )

# print("Aircraft loaded successfully!")
# print(f"Aircraft type: {AC.acName}")

# # 查看对象有哪些属性
# print("\nAvailable attributes:")
# attributes = [attr for attr in dir(AC) if not attr.startswith('_')]
# for attr in sorted(attributes)[:10]:  # 显示前10个属性
#     print(f"  {attr}")

# # 或者查看所有以 'bada' 开头的属性
# bada_attrs = [attr for attr in dir(AC) if 'bada' in attr.lower()]
# print(f"\nBADA related attributes: {bada_attrs}")

# # 尝试其他可能的版本属性
# possible_version_attrs = ['version', 'badaVersion', 'bada_version', 'modelVersion']
# for attr in possible_version_attrs:
#     if hasattr(AC, attr):
#         print(f"Found version attribute '{attr}': {getattr(AC, attr)}")
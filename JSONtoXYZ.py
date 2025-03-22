import json

# JSON 파일 열기
with open('C:/Users/user/Downloads/2020mltermproject3dclassification/data.json', 'r') as f:
    data = json.load(f)

# 'train' 리스트에서 'points' 가져오기
points = data['train'][0]['points']

# .xyz 파일로 저장
with open('output.xyz', 'w') as f: 
    for point in points: 
        f.write(f"{point[0]} {point[1]} {point[2]}\n")

print("변환 완료! output.xyz 생성됨")

with open('../storage/countries_.txt', 'r') as f:
    lines = f.readlines()
lines = [l.strip() for l in lines]
countries = []
for l in range(0, len(lines), 3):
    countries.append(lines[l])
print(len(countries))
countries = [f"{c}\n" for c in countries]
with open('../storage/countries.txt', 'w') as f:
    f.writelines(countries)

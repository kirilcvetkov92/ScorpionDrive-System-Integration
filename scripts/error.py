import sys
import os
import csv

#          kp    ki   kd
# 15.5 - (0.25, 0.0, 3.0)
# 14.4 - (0.5,  0.0, 3.0)
# 13.7 - (0.8,  0.0, 3.0)
# 9.1  - (1.2,  0.0, 3.0)
# 7.8  - (1.2,  0.0, 2.0)
# 3.75 - (1.2, 0.0,  1.5)
# 1.2  - (1.2, 0.0,  1.0)
# ???? - (1.2, 0.0,  0.8)
# ???? - (0.0, 0.0, 0.0)

if __name__ == '__main__':
    usage = "%s file.csv" % os.path.basename(sys.argv[0])

    if len(sys.argv) != 2:
        sys.stderr.write("%s\n" % usage)
        sys.exit(1)

    csv_file = sys.argv[1]
    error = 0
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        for index, row in enumerate(reader):
            if index == 0:
                continue
            actual, proposed = float(row[0]), float(row[1])
            error += (actual - proposed)**2

    print(error)

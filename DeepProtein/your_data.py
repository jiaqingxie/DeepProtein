import random
import string
def random_protein_name(length=5):
    return ''.join(random.choices(string.ascii_uppercase, k=length))


def random_aim_value():
    return round(random.uniform(0, 100), 2)


def generate_protein_aim_pairs(count=10,
                               filename="protein_aim.csv",
                               sep=",",
                               with_two_proteins=False):
    with open(filename, "w", encoding="utf-8") as f:
        for _ in range(count):
            p1 = random_protein_name()
            aim = random_aim_value()
            if with_two_proteins:
                p2 = random_protein_name()
                line = f"{p1}{sep}{p2}{sep}{aim}\n"
            else:
                line = f"{p1}{sep}{aim}\n"
            f.write(line)

    print(f"Write {count} rows to the file {filename}，format=with_two_proteins={with_two_proteins}.")


def read_protein_aim_file(filename, sep=","):
    data_list = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(sep)
            parts = [p.strip() for p in parts if p]

            if len(parts) == 2:
                p1, aim = parts
                data_list.append((p1, aim))
            elif len(parts) == 3:
                p1, p2, aim = parts
                data_list.append((p1, p2, aim))
            else:
                print(f"Warning: {line}")

    return data_list


def split_data(data, ratio=(0.5, 0.3, 0.2), shuffle=True):
    if shuffle:
        random.shuffle(data)

    total = len(data)
    train_end = int(total * ratio[0])
    valid_end = train_end + int(total * ratio[1])

    train_data = data[:train_end]
    valid_data = data[train_end:valid_end]
    test_data = data[valid_end:]
    return train_data, valid_data, test_data


def demo():
    generate_protein_aim_pairs(count=10,
                               filename="protein_aim.csv",
                               sep=",",
                               with_two_proteins=False)
    data_2col = read_protein_aim_file("protein_aim.csv", sep=",")

    train_data, valid_data, test_data = split_data(data_2col, ratio=(0.5, 0.3, 0.2), shuffle=True)

    train_zipped = list(zip(*train_data))
    valid_zipped = list(zip(*valid_data))
    test_zipped = list(zip(*test_data))

    print("\n--- (train_data) ---")
    print("Initial:", train_data)
    print("After zipping:", train_zipped)

    print("\n--- (valid_data) ---")
    print("Initial:", valid_data)
    print("After zipping:", valid_zipped)

    print("\n--- (test_data) ---")
    print("Initial:", test_data)
    print("After zipping:", test_zipped)


# if __name__ == "__main__":
#     demo()

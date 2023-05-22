# ~/~ begin <<docs/l-systems.md#demo/sierspinsky_table.py>>[init]
from .lsystem import sierspinsky

if __name__ == "__main__":
    print("| generation | string | size |")
    print("| ---------- | ------ | ----:|")

    for i in range(7):
        gen = "".join(sierspinsky.expand(i))
        size = len(gen)
        gen_short = gen if size < 50 else gen[:24] + " ... " + gen[-24:]
        print(f"| {i} | `{gen_short}` | {size} |")
# ~/~ end
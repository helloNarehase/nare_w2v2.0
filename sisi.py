from time import sleep

COLORS = [
    "\033[91m",
    "\033[93m",
    "\033[92m",
    "\033[96m",
    "\033[94m",
    "\033[95m",
    "\033[97m"
]
RESET = "\033[0m"

def print_towers(tower_a, tower_b, tower_c, total):
    print("\033[2J\033[H")
    width = 2 * total - 1
    print((("+" + "-" * ((width + 1) - 1)) * 3) + "+")
    for row in range(total):
        line = ""
        for tower in (tower_a, tower_b, tower_c):
            tower_reversed = tower[::-1]
            blank_count = total - len(tower)
            if row < blank_count:
                disk_str = " " * width
            else:
                disk = tower_reversed[row - blank_count]
                color = COLORS[(disk - 1) % len(COLORS)]
                disk_str = color + ("+" * (2 * disk - 1)).center(width, " ") + RESET
            line += "|" + disk_str
        line += "|"
        print(line)
    print((("+" + "-" * ((width + 1) - 1)) * 3) + "+")
    sleep(0.5)

def hanoi(n, src, aux, dest, towers, total):
    if n == 1:
        disk = towers[src].pop()
        towers[dest].append(disk)
        print(f"\nMove disk {disk} from Tower {src+1} to Tower {dest+1}")
        print_towers(towers[0], towers[1], towers[2], total)
    else:
        hanoi(n - 1, src, dest, aux, towers, total)
        disk = towers[src].pop()
        towers[dest].append(disk)
        print(f"\nMove disk {disk} from Tower {src+1} to Tower {dest+1}")
        print_towers(towers[0], towers[1], towers[2], total)
        hanoi(n - 1, aux, src, dest, towers, total)

n_disks = 8
towers = [[], list(range(n_disks, 0, -1)), [],]

print("Initial state:")
print_towers(towers[0], towers[1], towers[2], n_disks)
hanoi(n_disks, 1, 0, 2, towers, n_disks)
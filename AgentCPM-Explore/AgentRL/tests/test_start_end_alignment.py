from collections import Counter
import random

# ========== 被测试函数 ==========
def align_with_constant_gap(start_indices, end_indices):
    starts = sorted(list(start_indices))
    ends = sorted(list(end_indices))

    # 推断 gap
    gaps = []
    si = 0
    for e in ends:
        while si < len(starts) and starts[si] <= e:
            si += 1
        if si < len(starts):
            gaps.append(starts[si] - e)

    G = Counter(gaps).most_common(1)[0][0] if gaps else None

    aligned_s = []
    aligned_e = []

    i, j = 0, 0
    while i < len(starts) and j < len(ends):
        s = starts[i]
        if ends[j] <= s:
            j += 1
            continue
        e = ends[j]

        ok = False
        if i + 1 < len(starts):
            if G is None or (starts[i+1] - e == G):
                ok = True
        else:
            ok = True

        if ok:
            aligned_s.append(s)
            aligned_e.append(e)
            i += 1
            j += 1
        else:
            j += 1

    return aligned_s, aligned_e



# ========== 工具函数 ==========
def check_result(s_out, e_out):
    if len(s_out) != len(e_out):
        return False, "length mismatch"

    for s, e in zip(s_out, e_out):
        if not (s < e):
            return False, f"invalid pair: ({s}, {e})"
    return True, "OK"


# ========== 测试用例 ==========
def test_case(starts, ends, description):
    print(f"\n===== {description} =====")
    print("input starts:", starts)
    print("input ends:  ", ends)

    s_out, e_out = align_with_constant_gap(starts, ends)

    print("aligned starts:", s_out)
    print("aligned ends:  ", e_out)

    ok, msg = check_result(s_out, e_out)
    print("check:", msg)


# 1. 完全正常情况
test_case([10, 30, 50], [15, 35, 55], "正常情况")

# 2. 中间有 mismatch（测试 gap 校验是否能跳过错误的 end）
test_case([10, 30, 50], [15, 100, 55], "中间 mismatch")

# 3. 末尾 mismatch
test_case([10, 30, 50], [15, 35, 999], "末尾 mismatch")

# 4. 多余的 start
test_case([10, 30, 50, 100], [15, 35, 55], "多余 start")

# 5. 多余的 end
test_case([10, 30, 50], [15, 35, 55, 300], "多余 end")

# 6. 完全无法对应
test_case([100, 200], [5, 10], "完全不可能匹配")

# 7. 只有一个 start
test_case([10], [20], "单个 start")

# 8. 只有一个 end
test_case([5], [10, 20], "单个 end")

# 9. start 和 end 混乱顺序
test_case([50, 10, 30], [35, 15, 55], "乱序")

# 10. gap 噪声（中间插入错误 end）
test_case([10, 30, 50], [15, 17, 35, 55], "带随机噪声")



# ========== 随机压力测试（Fuzz） ==========
def random_test(n=1000):
    for _ in range(n):
        # 随机生成 gap = 常量
        gap = random.randint(3, 100)

        # 随机生成 start
        starts = sorted(random.sample(range(0, 2000, 5), random.randint(1, 10)))

        # 按 gap 生成正确 end
        ends = [s + random.randint(1, 10) for s in starts]

        # 随机制造噪声
        if random.random() < 0.4:  # 加噪声
            noise = random.sample(range(0, 2000), random.randint(0, 3))
            ends.extend(noise)

        random.shuffle(ends)
        s_out, e_out = align_with_constant_gap(starts, ends)
        ok, msg = check_result(s_out, e_out)
        if not ok:
            print("Fuzz test failed!!!")
            print("starts:", starts)
            print("ends:", ends)
            print("aligned:", list(zip(s_out, e_out)))
            print("reason:", msg)
            return
    print(f"Fuzz test({n}) passed without errors.")


random_test(500)

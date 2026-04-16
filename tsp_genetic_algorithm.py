import numpy as np
import matplotlib.pyplot as plt
import time
from typing import List, Tuple, Optional

# ============================================================
# 0. 初始参数配置（从外部配置文件导入，修改参数在 config_GA.py）
# ============================================================
from config_GA import (
    SEED, 
    N_CITIES, 
    COORD_RANGE, 
    ID_LAST2,
    INITIAL_POP_METHOD, 
    POP_SIZE, 
    N_GENERATIONS,
    CROSSOVER_PROB, 
    MUTATION_PROB, 
    ELITE_SIZE, 
    PATIENCE,
    SELECTION_METHOD, 
    TOURNAMENT_SIZE, 
    CROSSOVER_METHOD, 
    MUTATION_METHOD,
    POP_SIZES, 
    CROSSOVER_PROBS, 
    SELECTION_METHODS,
)

# 配置中文字体（用于画图）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 1. 城市坐标生成
# ============================================================
def generate_cities(seed: int, n: int = 50) -> np.ndarray:
    """
    使用与原程序完全一致的方式生成城市坐标。
    坐标范围：[0, 1000] x [0, 1000]
    """
    np.random.seed(seed)
    x = np.random.uniform(0, COORD_RANGE, n)
    y = np.random.uniform(0, COORD_RANGE, n)
    return np.column_stack((x, y))  # 返回城市坐标数组


# ============================================================
# 2. 城市距离计算（欧式距离）
# ============================================================
def calc_distance_matrix(cities: np.ndarray) -> np.ndarray:
    """
    计算城市间欧氏距离矩阵
    输入：
        cities: 城市坐标数组
    输出：
        dist_matrix: 城市间欧氏距离矩阵
    """
    diff = cities[:, np.newaxis, :] - cities[np.newaxis, :, :]  # 利用广播机制计算两两城市间的坐标差
    return np.sqrt((diff ** 2).sum(axis=2))  # 对最后一轴求平方和再开根，得到欧氏距离矩阵


def tour_length(tour: List[int], dist_matrix: np.ndarray) -> float:
    """
    计算路径总长度
    输入：
        tour: 路径
        dist_matrix: 城市间欧氏距离矩阵
    输出：
        total: 路径总长度
    """
    total = 0.0
    n = len(tour)
    for i in range(n):  # 遍历路径中的每个城市
        total += dist_matrix[tour[i]][tour[(i + 1) % n]]  # 让路径首尾相连，形成闭合回路
    return total


# ============================================================
# 3. 初始个体与种群生成
# ============================================================
def generate_initial_tour_nearest_neighbor(dist_matrix: np.ndarray, start_city: Optional[int] = None) -> List[int]:
    """
    最近邻（Nearest Neighbor）贪心初解：
      从 start_city 出发，每一步都去当前城市最近且未访问的城市。
      start_city 不传时，会使用随机起点（保证可复现）。
    输入：
        dist_matrix: 城市间欧氏距离矩阵
        start_city: 初始起点城市
    输出：
        tour: 最邻近贪心初始路径
    """
    n = dist_matrix.shape[0]
    start_city = int(np.random.randint(n)) if start_city is None else int(start_city)
    visited = np.zeros(n, dtype=bool)  # 访问标记数组：True 表示城市已被放入路径
    visited[start_city] = True  # 起点标记为已访问
    tour = [start_city]  # 用列表保存路径顺序（从起点开始）
    for _ in range(n - 1):  # 除起点外还需要选择 n-1 次后继城市
        current = tour[-1]  # 当前所在城市编号（即路径最后一个元素）
        candidates = np.where(~visited)[0]  # 所有未访问城市编号的候选集合
        next_city = int(candidates[int(np.argmin(dist_matrix[current][candidates]))])  # 选取最近的未访问城市
        tour.append(next_city)  # 把最近城市加入路径末尾
        visited[next_city] = True  # 将该城市标记为已访问
    return tour


def generate_individual(dist_matrix: np.ndarray, init_method: str) -> List[int]:
    """
    个体编码方式：采用城市编号的排列编码。
    输入：
        dist_matrix: 城市间距离矩阵
        init_method: 初始个体生成方式
    输出：
        individual: 一个合法 TSP 个体
    """
    n = dist_matrix.shape[0]
    if init_method == 'random':
        individual = list(range(n))
        np.random.shuffle(individual)
        return individual
    if init_method == 'nearest_neighbor':
        return generate_initial_tour_nearest_neighbor(dist_matrix)
    raise ValueError(f'Invalid init_method: {init_method}')


def initialize_population(pop_size: int, dist_matrix: np.ndarray, init_method: str) -> List[List[int]]:
    """
    初始化种群
    输入：
        pop_size: 种群规模
        dist_matrix: 城市间距离矩阵
        init_method: 初始个体生成方式
    输出：
        population: 初始种群
    """
    return [generate_individual(dist_matrix, init_method) for _ in range(pop_size)]


# ============================================================
# 4. 适应度函数
# ============================================================
def evaluate_population(population: List[List[int]], dist_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算整个种群的路径长度与适应度。
    适应度函数：fitness = 1 / (length + 1e-9)
    输入：
        population: 当前种群
        dist_matrix: 城市间距离矩阵
    输出：
        lengths: 各个个体的路径长度
        fitness: 各个个体的适应度
    """
    lengths = np.array([tour_length(ind, dist_matrix) for ind in population], dtype=float)
    fitness = 1.0 / (lengths + 1e-9)
    return lengths, fitness


# ============================================================
# 5. 选择操作（三种实现方式：轮盘赌、锦标赛、排名选择）
# ============================================================
def roulette_selection(population: List[List[int]], fitness: np.ndarray) -> List[int]:
    """
    轮盘赌选择：按照个体适应度占比随机抽取父代。
    输入：
        population: 当前种群
        fitness: 种群适应度数组
    输出：
        parent: 被选中的父代个体
    """
    idx = int(np.random.choice(len(population), p=fitness / fitness.sum()))
    return population[idx].copy()


def tournament_selection(population: List[List[int]], fitness: np.ndarray, k: int) -> List[int]:
    """
    锦标赛选择：随机抽取 k 个个体，返回其中适应度最高者。
    输入：
        population: 当前种群
        fitness: 种群适应度数组
        k: 锦标赛规模
    输出：
        parent: 被选中的父代个体
    """
    k = min(k, len(population))
    idx = np.random.choice(len(population), size=k, replace=False)
    return population[int(idx[np.argmax(fitness[idx])])].copy()


def rank_selection(population: List[List[int]], fitness: np.ndarray) -> List[int]:
    """
    排名选择：根据适应度排序后分配选择概率。
    输入：
        population: 当前种群
        fitness: 种群适应度数组
    输出：
        parent: 被选中的父代个体
    """
    order = np.argsort(fitness)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(population) + 1)
    idx = int(np.random.choice(len(population), p=ranks / ranks.sum()))
    return population[idx].copy()


def select_parent(population: List[List[int]], fitness: np.ndarray, method: str) -> List[int]:
    """
    按指定策略选择父代个体。
    输入：
        population: 当前种群
        fitness: 种群适应度数组
        method: 选择策略
    输出：
        parent: 被选中的父代个体
    """
    if method == 'roulette':
        return roulette_selection(population, fitness)
    if method == 'tournament':
        return tournament_selection(population, fitness, TOURNAMENT_SIZE)
    if method == 'rank':
        return rank_selection(population, fitness)
    raise ValueError(f'Invalid selection method: {method}')


# ============================================================
# 6. 交叉操作（顺序交叉 OX）
# ============================================================
def ordered_crossover(parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
    """
    顺序交叉（Ordered Crossover, OX）：
      随机选择一段区间保留父代片段，再按顺序填充剩余城市。
    输入：
        parent1: 父代个体 1
        parent2: 父代个体 2
    输出：
        child1: 子代个体 1
        child2: 子代个体 2
    """
    n = len(parent1)
    left, right = sorted(np.random.choice(n, 2, replace=False))
    child1 = [-1] * n
    child2 = [-1] * n
    child1[left:right + 1] = parent1[left:right + 1]
    child2[left:right + 1] = parent2[left:right + 1]
    remain1 = [city for city in parent2 if city not in child1]
    remain2 = [city for city in parent1 if city not in child2]
    for i, city in zip([i for i, v in enumerate(child1) if v == -1], remain1):
        child1[i] = city
    for i, city in zip([i for i, v in enumerate(child2) if v == -1], remain2):
        child2[i] = city
    return child1, child2


def crossover(parent1: List[int], parent2: List[int], prob: float, method: str) -> Tuple[List[int], List[int]]:
    """
    按给定交叉概率执行交叉操作。
    输入：
        parent1: 父代个体 1
        parent2: 父代个体 2
        prob: 交叉概率
        method: 交叉方式
    输出：
        child1: 子代个体 1
        child2: 子代个体 2
    """
    if np.random.random() >= prob:
        return parent1.copy(), parent2.copy()
    if method == 'ordered':
        return ordered_crossover(parent1, parent2)
    raise ValueError(f'Invalid crossover method: {method}')


# ============================================================
# 7. 变异操作（三种实现方式：交换、插入、逆转）
# ============================================================
def mutate_swap(individual: List[int]) -> List[int]:
    """
    swap 变异：随机交换两个城市的位置。
    输入：
        individual: 当前个体
    输出：
        mutant: 变异后的个体
    """
    mutant = individual.copy()
    i, j = np.random.choice(len(mutant), 2, replace=False)
    mutant[i], mutant[j] = mutant[j], mutant[i]
    return mutant


def mutate_insert(individual: List[int]) -> List[int]:
    """
    insert 变异：随机取出一个城市并插入到新位置。
    输入：
        individual: 当前个体
    输出：
        mutant: 变异后的个体
    """
    mutant = individual.copy()
    city = mutant.pop(int(np.random.randint(len(mutant))))
    mutant.insert(int(np.random.randint(len(mutant) + 1)), city)
    return mutant


def mutate_reverse(individual: List[int]) -> List[int]:
    """
    reverse 变异：随机选择一段子序列并逆转。
    输入：
        individual: 当前个体
    输出：
        mutant: 变异后的个体
    """
    mutant = individual.copy()
    i, j = sorted(np.random.choice(len(mutant), 2, replace=False))
    mutant[i:j + 1] = mutant[i:j + 1][::-1]
    return mutant


def mutate(individual: List[int], prob: float, method: str) -> List[int]:
    """
    按给定变异概率执行变异操作。
    输入：
        individual: 当前个体
        prob: 变异概率
        method: 变异方式
    输出：
        mutant: 变异后的个体
    """
    if np.random.random() >= prob:
        return individual.copy()
    if method == 'swap':
        return mutate_swap(individual)
    if method == 'insert':
        return mutate_insert(individual)
    if method == 'reverse':
        return mutate_reverse(individual)
    raise ValueError(f'Invalid mutation method: {method}')


# ============================================================
# 8. 遗传算法核心求解函数
# ============================================================
def genetic_algorithm(
    cities: np.ndarray,
    dist_matrix: np.ndarray,
    pop_size: int = POP_SIZE,
    n_generations: int = N_GENERATIONS,
    crossover_prob: float = CROSSOVER_PROB,
    mutation_prob: float = MUTATION_PROB,
    elite_size: int = ELITE_SIZE,
    patience: int = PATIENCE,
    selection_method: str = SELECTION_METHOD,
    init_method: str = INITIAL_POP_METHOD,
    crossover_method: str = CROSSOVER_METHOD,
    mutation_method: str = MUTATION_METHOD,
    rng_seed: int = SEED,
) -> Tuple[List[int], float, dict]:
    """
    遗传算法求解 TSP。

    输入：
        cities: 城市坐标数组
        dist_matrix: 距离矩阵
        pop_size: 种群规模
        n_generations: 最大迭代代数
        crossover_prob: 交叉概率
        mutation_prob: 变异概率
        elite_size: 精英保留数量
        patience: 提前停止阈值
        selection_method: 选择策略
        init_method: 初始种群生成方式
        crossover_method: 交叉方式
        mutation_method: 变异方式
        rng_seed: 随机种子

    输出：
        best_tour: 最优路径（城市索引列表）
        best_len: 最优路径长度
        history: 优化过程记录字典
    """
    np.random.seed(rng_seed)
    _ = cities

    # 初始种群：按 init_method 生成
    population = initialize_population(pop_size, dist_matrix, init_method)
    lengths, fitness = evaluate_population(population, dist_matrix)

    best_idx = int(np.argmin(lengths))
    best_tour = population[best_idx].copy()
    best_len = float(lengths[best_idx])
    no_improve_count = 0

    history = {
        'best_lengths': [],
        'avg_lengths': [],
        'avg_fitness': [],
        'generations': [],
    }

    # 外循环：按代更新种群
    for generation in range(1, n_generations + 1):
        elite_idx = np.argsort(lengths)[:elite_size]
        new_population = [population[i].copy() for i in elite_idx]  # 精英保留

        # 其余个体由选择、交叉、变异生成
        while len(new_population) < pop_size:
            parent1 = select_parent(population, fitness, selection_method)
            parent2 = select_parent(population, fitness, selection_method)
            child1, child2 = crossover(parent1, parent2, crossover_prob, crossover_method)
            new_population.append(mutate(child1, mutation_prob, mutation_method))
            if len(new_population) < pop_size:
                new_population.append(mutate(child2, mutation_prob, mutation_method))

        population = new_population
        lengths, fitness = evaluate_population(population, dist_matrix)
        current_best_idx = int(np.argmin(lengths))
        current_best_len = float(lengths[current_best_idx])

        # 更新全局最优解
        if current_best_len < best_len:
            best_len = current_best_len
            best_tour = population[current_best_idx].copy()
            no_improve_count = 0
        else:
            no_improve_count += 1

        # 记录本代数据
        history['best_lengths'].append(best_len)
        history['avg_lengths'].append(float(np.mean(lengths)))
        history['avg_fitness'].append(float(np.mean(fitness)))
        history['generations'].append(generation)

        # 连续若干代没有改进则提前停止
        if no_improve_count >= patience:
            break

    return best_tour, best_len, history


# ============================================================
# 9. 单组实验执行函数
# ============================================================
def run_case(cities: np.ndarray, dist_matrix: np.ndarray, **kwargs) -> dict:
    """
    运行一组遗传算法实验并返回结果。
    输入：
        cities: 城市坐标数组
        dist_matrix: 城市间距离矩阵
        **kwargs: 遗传算法附加参数
    输出：
        result: 单组实验结果字典
    """
    t_start = time.time()
    best_tour, best_len, history = genetic_algorithm(cities, dist_matrix, **kwargs)
    return {
        'best_tour': best_tour,
        'best_len': best_len,
        'generations': history['generations'][-1],
        'elapsed': time.time() - t_start,
        'history': history,
    }


# ============================================================
# 10. 可视化：路线图 + 收敛曲线
# ============================================================
def plot_single_result(
    cities: np.ndarray,
    tour: List[int],
    history: dict,
    best_len: float,
    method_label: str = '基准参数',
    save_path: str = None
):
    """
    绘制单次实验结果（2x2 四子图）：
      1) 最优路径图
      2) 最优路径长度变化
      3) 平均路径长度变化
      4) 平均适应度变化
    输入：
        cities: 城市坐标数组
        tour: 路径
        history: 优化过程记录字典
        best_len: 最优路径长度
        method_label: 该最优结果对应的方法说明
        save_path: 保存路径
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        f'遗传算法 TSP  |  最优路径长度: {best_len:.2f}  |  方法: {method_label}',
        fontsize=14, fontweight='bold'
    )

    ax1 = axes[0, 0]  # 最优路径
    ax2 = axes[0, 1]  # 最优路径长度变化
    ax3 = axes[1, 0]  # 平均路径长度变化
    ax4 = axes[1, 1]  # 平均适应度变化

    # 图1：最优路径图
    tour_closed = tour + [tour[0]]
    coords = cities[tour_closed]
    ax1.plot(coords[:, 0], coords[:, 1], 'b-', linewidth=1.2, alpha=0.7, label='路线')
    ax1.scatter(cities[:, 0], cities[:, 1], c='tomato', s=60, zorder=5)
    ax1.scatter(cities[tour[0], 0], cities[tour[0], 1],
                c='limegreen', s=160, marker='*', zorder=6, label='起点')
    for idx, (x, y) in enumerate(cities):
        ax1.annotate(str(idx), (x, y), fontsize=6,
                     ha='center', va='center', color='black')
    ax1.set_title('最优路径图', fontsize=12)
    ax1.set_xlabel('X 坐标')
    ax1.set_ylabel('Y 坐标')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-20, COORD_RANGE + 20)
    ax1.set_ylim(-20, COORD_RANGE + 20)

    # 图2：最优路径长度变化
    gens = history['generations']
    ax2.plot(gens, history['best_lengths'], 'b-', linewidth=2, label='最优路径长度')
    ax2.set_title('最优路径长度变化', fontsize=12)
    ax2.set_xlabel('迭代代数')
    ax2.set_ylabel('路径长度')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # 图3：平均路径长度变化
    ax3.plot(gens, history['avg_lengths'], 'r-', linewidth=2, label='平均路径长度')
    ax3.set_title('平均路径长度变化', fontsize=12)
    ax3.set_xlabel('迭代代数')
    ax3.set_ylabel('路径长度')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    # 图4：平均适应度变化
    ax4.plot(gens, history['avg_fitness'], color='#2ca02c', linewidth=2, label='平均适应度')
    ax4.set_title('平均适应度变化', fontsize=12)
    ax4.set_xlabel('迭代代数')
    ax4.set_ylabel('平均适应度')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'  图表已保存：{save_path}')
    plt.close(fig)


# ============================================================
# 11. 可视化：参数对比图
# ============================================================
def plot_comparison(results: List[dict], labels: List[str], title: str, save_path: str):
    """
    绘制三组参数的对比图（2x2 四子图）：
      1) 最优路径长度对比
      2) 平均路径长度对比
      3) 迭代代数对比
      4) 耗时对比
    输入：
        results: 实验结果列表
        labels: 横轴标签
        title: 图标题
        save_path: 保存路径
    """
    colors = ['#e74c3c', '#2ecc71', '#3498db']

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    ax1 = axes[0, 0]  # 最优路径长度对比
    ax2 = axes[0, 1]  # 平均路径长度对比
    ax3 = axes[1, 0]  # 迭代代数对比
    ax4 = axes[1, 1]  # 耗时对比

    best_lengths = [item['best_len'] for item in results]
    avg_lengths = [item['history']['avg_lengths'][-1] for item in results]
    generations = [item['generations'] for item in results]
    elapsed = [item['elapsed'] for item in results]

    ax1.bar(labels, best_lengths, color=colors)
    ax1.set_title('最优路径长度对比', fontsize=12)
    ax1.set_ylabel('最优路径长度')
    ax1.grid(True, axis='y', alpha=0.3)

    ax2.bar(labels, avg_lengths, color=colors)
    ax2.set_title('平均路径长度对比', fontsize=12)
    ax2.set_ylabel('平均路径长度')
    ax2.grid(True, axis='y', alpha=0.3)

    ax3.bar(labels, generations, color=colors)
    ax3.set_title('迭代代数对比', fontsize=12)
    ax3.set_ylabel('迭代代数')
    ax3.grid(True, axis='y', alpha=0.3)

    ax4.bar(labels, elapsed, color=colors)
    ax4.set_title('耗时对比', fontsize=12)
    ax4.set_ylabel('耗时（秒）')
    ax4.grid(True, axis='y', alpha=0.3)

    for ax, values in zip([ax1, ax2, ax3, ax4], [best_lengths, avg_lengths, generations, elapsed]):
        for i, value in enumerate(values):
            text = f'{value:.2f}' if isinstance(value, float) else str(value)
            ax.text(i, value, text, ha='center', va='bottom', fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'对比图已保存：{save_path}')
    plt.close(fig)


# ============================================================
# 12. 结果表格打印
# ============================================================
def print_table(title: str, labels: List[str], results: List[dict]):
    """
    按表格形式打印实验结果。
    输入：
        title: 表标题
        labels: 参数标签列表
        results: 实验结果列表
    """
    print('\n' + '=' * 65)
    print(f'  {title}')
    print('=' * 65)
    print(f'  {"参数":<18} {"迭代代数":<16} {"最优路径长度":<18} {"耗时（秒）"}')
    print('  ' + '-' * 60)
    for label, item in zip(labels, results):
        print(f'  {label:<18} {item["generations"]:<16} {item["best_len"]:<18.2f} {item["elapsed"]:.2f}')


# ============================================================
# 13. 主程序入口
# ============================================================
if __name__ == '__main__':
    print('=' * 65)
    print('  TSP 遗传算法实验')
    print('  本轮实验参数如下：\n')
    print('  学号：125130024341')
    print(f'  随机种子 seed(学号后5位) = {SEED}')
    print(f'  城市数量 N = {N_CITIES}   坐标范围: {COORD_RANGE}x{COORD_RANGE}')
    print(f'  学号后两位 = {ID_LAST2}')
    print(f'  种群规模 POP_SIZE = {POP_SIZE}')
    print(f'  最大迭代代数 N_GENERATIONS = {N_GENERATIONS}')
    print(f'  交叉概率 CROSSOVER_PROB = {CROSSOVER_PROB}')
    print(f'  变异概率 MUTATION_PROB = {MUTATION_PROB}')
    print(f'  精英保留数 ELITE_SIZE = {ELITE_SIZE}')
    print(f'  选择策略   = {SELECTION_METHOD}')
    print(f'  初始种群方式 = {INITIAL_POP_METHOD}')
    print(f'  交叉方式   = {CROSSOVER_METHOD}')
    print(f'  变异方式   = {MUTATION_METHOD}')
    print(f'  patience(提前停止) = {PATIENCE}')
    print('=' * 65)

    # 生成个性化城市坐标
    cities = generate_cities(SEED, N_CITIES)
    dist_matrix = calc_distance_matrix(cities)

    print('\n城市坐标生成完毕（前5座城市）：')
    for i in range(5):
        print(f'  城市 {i:2d}: X={cities[i, 0]:.2f}  Y={cities[i, 1]:.2f}')

    # 基准参数实验
    print('\n' + '-' * 65)
    print('  开始基准参数实验')
    print('-' * 65)

    t_start = time.time()
    baseline = run_case(cities, dist_matrix)
    baseline_elapsed = time.time() - t_start

    print(f'  迭代代数       : {baseline["generations"]}')
    print(f'  最优路径长度   : {baseline["best_len"]:.2f}')
    print(f'  耗时           : {baseline_elapsed:.2f} 秒')

    # 三组参数对比实验
    print('\n' + '-' * 65)
    print(f'  开始参数对比实验（种群规模 = {" / ".join(map(str, POP_SIZES))}）')
    print('-' * 65)
    pop_results = []
    for pop_size in POP_SIZES:
        print(f'\n[ pop_size = {pop_size} ]')
        result = run_case(cities, dist_matrix, pop_size=pop_size)
        print(f'  迭代代数       : {result["generations"]}')
        print(f'  最优路径长度   : {result["best_len"]:.2f}')
        print(f'  耗时           : {result["elapsed"]:.2f} 秒')
        pop_results.append(result)

    print('\n' + '-' * 65)
    print(f'  开始参数对比实验（交叉概率 = {" / ".join(map(str, CROSSOVER_PROBS))}）')
    print('-' * 65)
    cross_results = []
    for crossover_prob in CROSSOVER_PROBS:
        print(f'\n[ crossover_prob = {crossover_prob} ]')
        result = run_case(cities, dist_matrix, crossover_prob=crossover_prob)
        print(f'  迭代代数       : {result["generations"]}')
        print(f'  最优路径长度   : {result["best_len"]:.2f}')
        print(f'  耗时           : {result["elapsed"]:.2f} 秒')
        cross_results.append(result)

    print('\n' + '-' * 65)
    print(f'  开始参数对比实验（选择策略 = {" / ".join(SELECTION_METHODS)}）')
    print('-' * 65)
    select_results = []
    for selection_method in SELECTION_METHODS:
        print(f'\n[ selection_method = {selection_method} ]')
        result = run_case(cities, dist_matrix, selection_method=selection_method)
        print(f'  迭代代数       : {result["generations"]}')
        print(f'  最优路径长度   : {result["best_len"]:.2f}')
        print(f'  耗时           : {result["elapsed"]:.2f} 秒')
        select_results.append(result)

    # 从基准实验与全部参数对比实验中，选出真正最短路径对应的方法
    all_results = [
        ('基准参数（selection=tournament, pop_size=120, crossover_prob=0.90）', baseline),
        *[(f'种群规模 pop_size={v}', result) for v, result in zip(POP_SIZES, pop_results)],
        *[(f'交叉概率 crossover_prob={v}', result) for v, result in zip(CROSSOVER_PROBS, cross_results)],
        *[(f'选择策略 selection_method={v}', result) for v, result in zip(SELECTION_METHODS, select_results)],
    ]
    best_method_label, overall_best = min(all_results, key=lambda item: item[1]['best_len'])

    print('\n' + '-' * 65)
    print('  生成全实验真正最短路径结果图...')
    print(f'  最优方法       : {best_method_label}')
    print(f'  最优路径长度   : {overall_best["best_len"]:.2f}')
    print('-' * 65)

    plot_single_result(
        cities,
        overall_best['best_tour'],
        overall_best['history'],
        overall_best['best_len'],
        best_method_label,
        'tsp_ga_best_result.png'
    )

    # 汇总对比图
    print('\n' + '-' * 65)
    print('  生成参数对比汇总图...')
    plot_comparison(pop_results, [str(v) for v in POP_SIZES], '遗传算法 TSP 的种群规模参数对比', 'ga_pop_size_comparison.png')
    plot_comparison(cross_results, [str(v) for v in CROSSOVER_PROBS], '遗传算法 TSP 的交叉概率参数对比', 'ga_crossover_prob_comparison.png')
    plot_comparison(select_results, SELECTION_METHODS, '遗传算法 TSP 的选择策略参数对比', 'ga_selection_method_comparison.png')

    # 打印结果汇总表
    print_table('种群规模参数对比', [str(v) for v in POP_SIZES], pop_results)
    print_table('交叉概率参数对比', [str(v) for v in CROSSOVER_PROBS], cross_results)
    print_table('选择策略参数对比', SELECTION_METHODS, select_results)

    print('\n' + '=' * 65)
    print('  实验结果汇总')
    print('=' * 65)
    print(f'  基准实验最优路径长度 : {baseline["best_len"]:.2f}')
    print(f'  基准实验迭代代数     : {baseline["generations"]}')
    print(f'  全实验最优方法       : {best_method_label}')
    print(f'  全实验最优路径长度   : {overall_best["best_len"]:.2f}')
    print(f'  全实验对应迭代代数   : {overall_best["generations"]}')
    print('  图表文件：tsp_ga_best_result.png / ga_pop_size_comparison.png /')
    print('           ga_crossover_prob_comparison.png / ga_selection_method_comparison.png')
    print('=' * 65)

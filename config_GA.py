# ============================================================
# TSP 遗传算法 —— 初始参数配置文件
# ============================================================
# 修改此文件中的参数即可调整实验设置，无需改动主程序逻辑。

# 随机种子（取学号后5位）
SEED: int = 24341

# 城市总数
N_CITIES: int = 50

# 坐标范围（生成 COORD_RANGE x COORD_RANGE 的平面）
COORD_RANGE: int = 1000

# 学号后两位（动态计算，用于结果展示）
ID_LAST2: int = SEED % 100

# 初始种群生成方法（可选：'random'、'nearest_neighbor'）
#   - 'random'：随机排列生成初始个体
#   - 'nearest_neighbor'：最邻近贪心生成初始个体
INITIAL_POP_METHOD: str = 'random'

# 基准种群规模
POP_SIZE: int = 120

# 最大迭代代数
N_GENERATIONS: int = 1000

# 交叉概率
CROSSOVER_PROB: float = 0.90

# 变异概率
MUTATION_PROB: float = 0.20

# 精英保留数
ELITE_SIZE: int = 2

# 提前停止超参数：连续若干代无最优解改进则停止
PATIENCE: int = 120

# 选择策略（可选：'roulette'、'tournament'、'rank'）
SELECTION_METHOD: str = 'tournament'

# 锦标赛规模（仅在 tournament 选择中生效）
TOURNAMENT_SIZE: int = 4

# 交叉方式（可选：'ordered'）
CROSSOVER_METHOD: str = 'ordered'

# 变异方式（可选：'swap'、'insert'、'reverse'）
MUTATION_METHOD: str = 'swap'

# 参数对比实验的种群规模列表
POP_SIZES: list = [60, 120, 180]

# 参数对比实验的交叉概率列表
CROSSOVER_PROBS: list = [0.70, 0.85, 0.95]

# 参数对比实验的选择策略列表
SELECTION_METHODS: list = ['roulette', 'tournament', 'rank']

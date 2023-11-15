import sys
from pathlib import Path

DEBUG = sys.gettrace() is not None


RESULTS_DIR = Path("results/CASAS")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
# EXP_DIR = RESULTS_DIR / ''.join(np.random.choice(
#     list(string.ascii_lowercase + string.digits),
#     size=6,
#     replace=True
# ))
# EXP_DIR.mkdir(exist_ok=False)
# print(f"Experiment directory: {EXP_DIR}")

# summary_file = open(EXP_DIR / "summary.txt", "w+")

RND_SEED = 42


sedentary_cols = ["Bed_Toilet_Transition", "Relax", "Sleep", "Work"]
active_cols = ["Cook", "Eat", "Enter_Home", "Leave_Home", "Personal_Hygiene", "Wash_Dishes"]
activity_cols = sedentary_cols + active_cols

datetimenm = "DateTime"
seccosnm = "SecondsCos"
secsinnm = "SecondsSin"
doycosnm = "DoyCos"
doysinnm = "DoySin"
idnm = "Person"
activitynm = "Activity"
agenm = "age"
age_grpnm = "age_group"
gendernm = "gender"
racenm = "race"
static_feats = [agenm, gendernm, racenm]
real_label = "Real"

sens1cols_og = ['Bathroom', 'Bedroom', 'DiningRoom', 'Hall', 'Kitchen', 'LivingRoom', 'Office',
             'OutsideDoor']
sens1cols = [col + "1" for col in sens1cols_og]

sens2cols_og = ['Bathroom', 'Bed', 'Bedroom', 'Chair', 'DiningRoom', 'Entry', 'FrontDoor', 'Hall',
             'Kitchen', 'LivingRoom', 'Office', 'OutsideDoor', 'Refrigerator', 'Sink', 'Stove',
             'Toilet']
sens2cols = [col + "2" for col in sens2cols_og]

time_cols = [seccosnm, secsinnm, doycosnm, doysinnm]

train_time_feats = time_cols + sens1cols + sens2cols
time_feats = train_time_feats + activity_cols

all_feats = time_feats + static_feats
all_train_feats = train_time_feats + static_feats
binary_label_col = "Sedentary"
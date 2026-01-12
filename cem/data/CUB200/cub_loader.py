"""
General utils for training, evaluation and data loading

Adapted from: https://github.com/yewsiang/ConceptBottleneck/blob/master/CUB/cub_loader.py
"""
import numpy as np
import os
import pickle
import torch
import torchvision.transforms as transforms

from collections import defaultdict
from pytorch_lightning import seed_everything

from PIL import Image
from torch.utils.data import Dataset, DataLoader

########################################################
## GENERAL DATASET GLOBAL VARIABLES
########################################################

N_CLASSES = 200


# IMPORANT NOTE: THIS PROCESSED DATASET NEEDS TO BE DOWNLOADED FIRST BEFORE
#                BEING ABLE TO RUN ANY CUB EXPERIMENTS!!
#                Instructions on how to download the pre-processed dataset can
#                be found in the original CBM paper's repository
#                found here: https://github.com/yewsiang/ConceptBottleneck
#                Specifically, here: https://github.com/yewsiang/ConceptBottleneck/blob/master/CUB/

# CAN BE OVERWRITTEN WITH AN ENV VARIABLE DATASET_DIR
DATASET_DIR = os.environ.get("DATASET_DIR", 'cem/data/CUB200/')


#########################################################
## CONCEPT INFORMATION REGARDING CUB
#########################################################

# CUB Class names

CLASS_NAMES = [
    "Black_footed_Albatross",
    "Laysan_Albatross",
    "Sooty_Albatross",
    "Groove_billed_Ani",
    "Crested_Auklet",
    "Least_Auklet",
    "Parakeet_Auklet",
    "Rhinoceros_Auklet",
    "Brewer_Blackbird",
    "Red_winged_Blackbird",
    "Rusty_Blackbird",
    "Yellow_headed_Blackbird",
    "Bobolink",
    "Indigo_Bunting",
    "Lazuli_Bunting",
    "Painted_Bunting",
    "Cardinal",
    "Spotted_Catbird",
    "Gray_Catbird",
    "Yellow_breasted_Chat",
    "Eastern_Towhee",
    "Chuck_will_Widow",
    "Brandt_Cormorant",
    "Red_faced_Cormorant",
    "Pelagic_Cormorant",
    "Bronzed_Cowbird",
    "Shiny_Cowbird",
    "Brown_Creeper",
    "American_Crow",
    "Fish_Crow",
    "Black_billed_Cuckoo",
    "Mangrove_Cuckoo",
    "Yellow_billed_Cuckoo",
    "Gray_crowned_Rosy_Finch",
    "Purple_Finch",
    "Northern_Flicker",
    "Acadian_Flycatcher",
    "Great_Crested_Flycatcher",
    "Least_Flycatcher",
    "Olive_sided_Flycatcher",
    "Scissor_tailed_Flycatcher",
    "Vermilion_Flycatcher",
    "Yellow_bellied_Flycatcher",
    "Frigatebird",
    "Northern_Fulmar",
    "Gadwall",
    "American_Goldfinch",
    "European_Goldfinch",
    "Boat_tailed_Grackle",
    "Eared_Grebe",
    "Horned_Grebe",
    "Pied_billed_Grebe",
    "Western_Grebe",
    "Blue_Grosbeak",
    "Evening_Grosbeak",
    "Pine_Grosbeak",
    "Rose_breasted_Grosbeak",
    "Pigeon_Guillemot",
    "California_Gull",
    "Glaucous_winged_Gull",
    "Heermann_Gull",
    "Herring_Gull",
    "Ivory_Gull",
    "Ring_billed_Gull",
    "Slaty_backed_Gull",
    "Western_Gull",
    "Anna_Hummingbird",
    "Ruby_throated_Hummingbird",
    "Rufous_Hummingbird",
    "Green_Violetear",
    "Long_tailed_Jaeger",
    "Pomarine_Jaeger",
    "Blue_Jay",
    "Florida_Jay",
    "Green_Jay",
    "Dark_eyed_Junco",
    "Tropical_Kingbird",
    "Gray_Kingbird",
    "Belted_Kingfisher",
    "Green_Kingfisher",
    "Pied_Kingfisher",
    "Ringed_Kingfisher",
    "White_breasted_Kingfisher",
    "Red_legged_Kittiwake",
    "Horned_Lark",
    "Pacific_Loon",
    "Mallard",
    "Western_Meadowlark",
    "Hooded_Merganser",
    "Red_breasted_Merganser",
    "Mockingbird",
    "Nighthawk",
    "Clark_Nutcracker",
    "White_breasted_Nuthatch",
    "Baltimore_Oriole",
    "Hooded_Oriole",
    "Orchard_Oriole",
    "Scott_Oriole",
    "Ovenbird",
    "Brown_Pelican",
    "White_Pelican",
    "Western_Wood_Pewee",
    "Sayornis",
    "American_Pipit",
    "Whip_poor_Will",
    "Horned_Puffin",
    "Common_Raven",
    "White_necked_Raven",
    "American_Redstart",
    "Geococcyx",
    "Loggerhead_Shrike",
    "Great_Grey_Shrike",
    "Baird_Sparrow",
    "Black_throated_Sparrow",
    "Brewer_Sparrow",
    "Chipping_Sparrow",
    "Clay_colored_Sparrow",
    "House_Sparrow",
    "Field_Sparrow",
    "Fox_Sparrow",
    "Grasshopper_Sparrow",
    "Harris_Sparrow",
    "Henslow_Sparrow",
    "Le_Conte_Sparrow",
    "Lincoln_Sparrow",
    "Nelson_Sharp_tailed_Sparrow",
    "Savannah_Sparrow",
    "Seaside_Sparrow",
    "Song_Sparrow",
    "Tree_Sparrow",
    "Vesper_Sparrow",
    "White_crowned_Sparrow",
    "White_throated_Sparrow",
    "Cape_Glossy_Starling",
    "Bank_Swallow",
    "Barn_Swallow",
    "Cliff_Swallow",
    "Tree_Swallow",
    "Scarlet_Tanager",
    "Summer_Tanager",
    "Artic_Tern",
    "Black_Tern",
    "Caspian_Tern",
    "Common_Tern",
    "Elegant_Tern",
    "Forsters_Tern",
    "Least_Tern",
    "Green_tailed_Towhee",
    "Brown_Thrasher",
    "Sage_Thrasher",
    "Black_capped_Vireo",
    "Blue_headed_Vireo",
    "Philadelphia_Vireo",
    "Red_eyed_Vireo",
    "Warbling_Vireo",
    "White_eyed_Vireo",
    "Yellow_throated_Vireo",
    "Bay_breasted_Warbler",
    "Black_and_white_Warbler",
    "Black_throated_Blue_Warbler",
    "Blue_winged_Warbler",
    "Canada_Warbler",
    "Cape_May_Warbler",
    "Cerulean_Warbler",
    "Chestnut_sided_Warbler",
    "Golden_winged_Warbler",
    "Hooded_Warbler",
    "Kentucky_Warbler",
    "Magnolia_Warbler",
    "Mourning_Warbler",
    "Myrtle_Warbler",
    "Nashville_Warbler",
    "Orange_crowned_Warbler",
    "Palm_Warbler",
    "Pine_Warbler",
    "Prairie_Warbler",
    "Prothonotary_Warbler",
    "Swainson_Warbler",
    "Tennessee_Warbler",
    "Wilson_Warbler",
    "Worm_eating_Warbler",
    "Yellow_Warbler",
    "Northern_Waterthrush",
    "Louisiana_Waterthrush",
    "Bohemian_Waxwing",
    "Cedar_Waxwing",
    "American_Three_toed_Woodpecker",
    "Pileated_Woodpecker",
    "Red_bellied_Woodpecker",
    "Red_cockaded_Woodpecker",
    "Red_headed_Woodpecker",
    "Downy_Woodpecker",
    "Bewick_Wren",
    "Cactus_Wren",
    "Carolina_Wren",
    "House_Wren",
    "Marsh_Wren",
    "Rock_Wren",
    "Winter_Wren",
    "Common_Yellowthroat",
]
# Set of CUB attributes selected by original CBM paper
SELECTED_CONCEPTS = [
    1,
    4,
    6,
    7,
    10,
    14,
    15,
    20,
    21,
    23,
    25,
    29,
    30,
    35,
    36,
    38,
    40,
    44,
    45,
    50,
    51,
    53,
    54,
    56,
    57,
    59,
    63,
    64,
    69,
    70,
    72,
    75,
    80,
    84,
    90,
    91,
    93,
    99,
    101,
    106,
    110,
    111,
    116,
    117,
    119,
    125,
    126,
    131,
    132,
    134,
    145,
    149,
    151,
    152,
    153,
    157,
    158,
    163,
    164,
    168,
    172,
    178,
    179,
    181,
    183,
    187,
    188,
    193,
    194,
    196,
    198,
    202,
    203,
    208,
    209,
    211,
    212,
    213,
    218,
    220,
    221,
    225,
    235,
    236,
    238,
    239,
    240,
    242,
    243,
    244,
    249,
    253,
    254,
    259,
    260,
    262,
    268,
    274,
    277,
    283,
    289,
    292,
    293,
    294,
    298,
    299,
    304,
    305,
    308,
    309,
    310,
    311,
]

# Names of all CUB attributes
CONCEPT_SEMANTICS = [
    "has_bill_shape::curved_(up_or_down)",
    "has_bill_shape::dagger",
    "has_bill_shape::hooked",
    "has_bill_shape::needle",
    "has_bill_shape::hooked_seabird",
    "has_bill_shape::spatulate",
    "has_bill_shape::all-purpose",
    "has_bill_shape::cone",
    "has_bill_shape::specialized",
    "has_wing_color::blue",
    "has_wing_color::brown",
    "has_wing_color::iridescent",
    "has_wing_color::purple",
    "has_wing_color::rufous",
    "has_wing_color::grey",
    "has_wing_color::yellow",
    "has_wing_color::olive",
    "has_wing_color::green",
    "has_wing_color::pink",
    "has_wing_color::orange",
    "has_wing_color::black",
    "has_wing_color::white",
    "has_wing_color::red",
    "has_wing_color::buff",
    "has_upperparts_color::blue",
    "has_upperparts_color::brown",
    "has_upperparts_color::iridescent",
    "has_upperparts_color::purple",
    "has_upperparts_color::rufous",
    "has_upperparts_color::grey",
    "has_upperparts_color::yellow",
    "has_upperparts_color::olive",
    "has_upperparts_color::green",
    "has_upperparts_color::pink",
    "has_upperparts_color::orange",
    "has_upperparts_color::black",
    "has_upperparts_color::white",
    "has_upperparts_color::red",
    "has_upperparts_color::buff",
    "has_underparts_color::blue",
    "has_underparts_color::brown",
    "has_underparts_color::iridescent",
    "has_underparts_color::purple",
    "has_underparts_color::rufous",
    "has_underparts_color::grey",
    "has_underparts_color::yellow",
    "has_underparts_color::olive",
    "has_underparts_color::green",
    "has_underparts_color::pink",
    "has_underparts_color::orange",
    "has_underparts_color::black",
    "has_underparts_color::white",
    "has_underparts_color::red",
    "has_underparts_color::buff",
    "has_breast_pattern::solid",
    "has_breast_pattern::spotted",
    "has_breast_pattern::striped",
    "has_breast_pattern::multi-colored",
    "has_back_color::blue",
    "has_back_color::brown",
    "has_back_color::iridescent",
    "has_back_color::purple",
    "has_back_color::rufous",
    "has_back_color::grey",
    "has_back_color::yellow",
    "has_back_color::olive",
    "has_back_color::green",
    "has_back_color::pink",
    "has_back_color::orange",
    "has_back_color::black",
    "has_back_color::white",
    "has_back_color::red",
    "has_back_color::buff",
    "has_tail_shape::forked_tail",
    "has_tail_shape::rounded_tail",
    "has_tail_shape::notched_tail",
    "has_tail_shape::fan-shaped_tail",
    "has_tail_shape::pointed_tail",
    "has_tail_shape::squared_tail",
    "has_upper_tail_color::blue",
    "has_upper_tail_color::brown",
    "has_upper_tail_color::iridescent",
    "has_upper_tail_color::purple",
    "has_upper_tail_color::rufous",
    "has_upper_tail_color::grey",
    "has_upper_tail_color::yellow",
    "has_upper_tail_color::olive",
    "has_upper_tail_color::green",
    "has_upper_tail_color::pink",
    "has_upper_tail_color::orange",
    "has_upper_tail_color::black",
    "has_upper_tail_color::white",
    "has_upper_tail_color::red",
    "has_upper_tail_color::buff",
    "has_head_pattern::spotted",
    "has_head_pattern::malar",
    "has_head_pattern::crested",
    "has_head_pattern::masked",
    "has_head_pattern::unique_pattern",
    "has_head_pattern::eyebrow",
    "has_head_pattern::eyering",
    "has_head_pattern::plain",
    "has_head_pattern::eyeline",
    "has_head_pattern::striped",
    "has_head_pattern::capped",
    "has_breast_color::blue",
    "has_breast_color::brown",
    "has_breast_color::iridescent",
    "has_breast_color::purple",
    "has_breast_color::rufous",
    "has_breast_color::grey",
    "has_breast_color::yellow",
    "has_breast_color::olive",
    "has_breast_color::green",
    "has_breast_color::pink",
    "has_breast_color::orange",
    "has_breast_color::black",
    "has_breast_color::white",
    "has_breast_color::red",
    "has_breast_color::buff",
    "has_throat_color::blue",
    "has_throat_color::brown",
    "has_throat_color::iridescent",
    "has_throat_color::purple",
    "has_throat_color::rufous",
    "has_throat_color::grey",
    "has_throat_color::yellow",
    "has_throat_color::olive",
    "has_throat_color::green",
    "has_throat_color::pink",
    "has_throat_color::orange",
    "has_throat_color::black",
    "has_throat_color::white",
    "has_throat_color::red",
    "has_throat_color::buff",
    "has_eye_color::blue",
    "has_eye_color::brown",
    "has_eye_color::purple",
    "has_eye_color::rufous",
    "has_eye_color::grey",
    "has_eye_color::yellow",
    "has_eye_color::olive",
    "has_eye_color::green",
    "has_eye_color::pink",
    "has_eye_color::orange",
    "has_eye_color::black",
    "has_eye_color::white",
    "has_eye_color::red",
    "has_eye_color::buff",
    "has_bill_length::about_the_same_as_head",
    "has_bill_length::longer_than_head",
    "has_bill_length::shorter_than_head",
    "has_forehead_color::blue",
    "has_forehead_color::brown",
    "has_forehead_color::iridescent",
    "has_forehead_color::purple",
    "has_forehead_color::rufous",
    "has_forehead_color::grey",
    "has_forehead_color::yellow",
    "has_forehead_color::olive",
    "has_forehead_color::green",
    "has_forehead_color::pink",
    "has_forehead_color::orange",
    "has_forehead_color::black",
    "has_forehead_color::white",
    "has_forehead_color::red",
    "has_forehead_color::buff",
    "has_under_tail_color::blue",
    "has_under_tail_color::brown",
    "has_under_tail_color::iridescent",
    "has_under_tail_color::purple",
    "has_under_tail_color::rufous",
    "has_under_tail_color::grey",
    "has_under_tail_color::yellow",
    "has_under_tail_color::olive",
    "has_under_tail_color::green",
    "has_under_tail_color::pink",
    "has_under_tail_color::orange",
    "has_under_tail_color::black",
    "has_under_tail_color::white",
    "has_under_tail_color::red",
    "has_under_tail_color::buff",
    "has_nape_color::blue",
    "has_nape_color::brown",
    "has_nape_color::iridescent",
    "has_nape_color::purple",
    "has_nape_color::rufous",
    "has_nape_color::grey",
    "has_nape_color::yellow",
    "has_nape_color::olive",
    "has_nape_color::green",
    "has_nape_color::pink",
    "has_nape_color::orange",
    "has_nape_color::black",
    "has_nape_color::white",
    "has_nape_color::red",
    "has_nape_color::buff",
    "has_belly_color::blue",
    "has_belly_color::brown",
    "has_belly_color::iridescent",
    "has_belly_color::purple",
    "has_belly_color::rufous",
    "has_belly_color::grey",
    "has_belly_color::yellow",
    "has_belly_color::olive",
    "has_belly_color::green",
    "has_belly_color::pink",
    "has_belly_color::orange",
    "has_belly_color::black",
    "has_belly_color::white",
    "has_belly_color::red",
    "has_belly_color::buff",
    "has_wing_shape::rounded-wings",
    "has_wing_shape::pointed-wings",
    "has_wing_shape::broad-wings",
    "has_wing_shape::tapered-wings",
    "has_wing_shape::long-wings",
    "has_size::large_(16_-_32_in)",
    "has_size::small_(5_-_9_in)",
    "has_size::very_large_(32_-_72_in)",
    "has_size::medium_(9_-_16_in)",
    "has_size::very_small_(3_-_5_in)",
    "has_shape::upright-perching_water-like",
    "has_shape::chicken-like-marsh",
    "has_shape::long-legged-like",
    "has_shape::duck-like",
    "has_shape::owl-like",
    "has_shape::gull-like",
    "has_shape::hummingbird-like",
    "has_shape::pigeon-like",
    "has_shape::tree-clinging-like",
    "has_shape::hawk-like",
    "has_shape::sandpiper-like",
    "has_shape::upland-ground-like",
    "has_shape::swallow-like",
    "has_shape::perching-like",
    "has_back_pattern::solid",
    "has_back_pattern::spotted",
    "has_back_pattern::striped",
    "has_back_pattern::multi-colored",
    "has_tail_pattern::solid",
    "has_tail_pattern::spotted",
    "has_tail_pattern::striped",
    "has_tail_pattern::multi-colored",
    "has_belly_pattern::solid",
    "has_belly_pattern::spotted",
    "has_belly_pattern::striped",
    "has_belly_pattern::multi-colored",
    "has_primary_color::blue",
    "has_primary_color::brown",
    "has_primary_color::iridescent",
    "has_primary_color::purple",
    "has_primary_color::rufous",
    "has_primary_color::grey",
    "has_primary_color::yellow",
    "has_primary_color::olive",
    "has_primary_color::green",
    "has_primary_color::pink",
    "has_primary_color::orange",
    "has_primary_color::black",
    "has_primary_color::white",
    "has_primary_color::red",
    "has_primary_color::buff",
    "has_leg_color::blue",
    "has_leg_color::brown",
    "has_leg_color::iridescent",
    "has_leg_color::purple",
    "has_leg_color::rufous",
    "has_leg_color::grey",
    "has_leg_color::yellow",
    "has_leg_color::olive",
    "has_leg_color::green",
    "has_leg_color::pink",
    "has_leg_color::orange",
    "has_leg_color::black",
    "has_leg_color::white",
    "has_leg_color::red",
    "has_leg_color::buff",
    "has_bill_color::blue",
    "has_bill_color::brown",
    "has_bill_color::iridescent",
    "has_bill_color::purple",
    "has_bill_color::rufous",
    "has_bill_color::grey",
    "has_bill_color::yellow",
    "has_bill_color::olive",
    "has_bill_color::green",
    "has_bill_color::pink",
    "has_bill_color::orange",
    "has_bill_color::black",
    "has_bill_color::white",
    "has_bill_color::red",
    "has_bill_color::buff",
    "has_crown_color::blue",
    "has_crown_color::brown",
    "has_crown_color::iridescent",
    "has_crown_color::purple",
    "has_crown_color::rufous",
    "has_crown_color::grey",
    "has_crown_color::yellow",
    "has_crown_color::olive",
    "has_crown_color::green",
    "has_crown_color::pink",
    "has_crown_color::orange",
    "has_crown_color::black",
    "has_crown_color::white",
    "has_crown_color::red",
    "has_crown_color::buff",
    "has_wing_pattern::solid",
    "has_wing_pattern::spotted",
    "has_wing_pattern::striped",
    "has_wing_pattern::multi-colored",
]

CONCEPT_SEMANTICS_SENTENCE = [
    "a bird with a curved bill",
    "a bird with a dagger bill",
    "a bird with a hooked bill",
    "a bird with a needle bill",
    "a seabird with a hooked bill",
    "a bird with a spatulate bill",
    "a bird with a all-purpose bill",
    "a bird with a cone bill",
    "a bird with a specialized bill",
    "a bird with blue wings",
    "a bird with brown wings",
    "a bird with iridescent wings",
    "a bird with purple wings",
    "a bird with rufous wings",
    "a bird with grey wings",
    "a bird with yellow wings",
    "a bird with olive wings",
    "a bird with green wings",
    "a bird with pink wings",
    "a bird with orange wings",
    "a bird with black wings",
    "a bird with white wings",
    "a bird with red wings",
    "a bird with buff wings",
    "a bird with a blue upperpart",
    "a bird with a brown upperpart",
    "a bird with an iridescent upperpart",
    "a bird with a purple upperpart",
    "a bird with a rufous upperpart",
    "a bird with a grey upperpart",
    "a bird with a yellow upperpart",
    "a bird with an olive upperpart",
    "a bird with a green upperpart",
    "a bird with a pink upperpart",
    "a bird with an orange upperpart",
    "a bird with a black upperpart",
    "a bird with a white upperpart",
    "a bird with a red upperpart",
    "a bird with a buff upperpart",
    "a bird with a blue underpart",
    "a bird with a brown underpart",
    "a bird with an iridescent underpart",
    "a bird with a purple underpart",
    "a bird with a rufous underpart",
    "a bird with a grey underpart",
    "a bird with a yellow underpart",
    "a bird with an olive underpart",
    "a bird with a green underpart",
    "a bird with a pink underpart",
    "a bird with an orange underpart",
    "a bird with a black underpart",
    "a bird with a white underpart",
    "a bird with a red underpart",
    "a bird with a buff underpart",
    "a bird with a solid breast pattern",
    "a bird with a spotted breast pattern",
    "a bird with a striped breast pattern",
    "a bird with a multi-colored breast pattern",
    "a bird with a blue back",
    "a bird with a brown back",
    "a bird with an iridescent back",
    "a bird with a purple back",
    "a bird with a rufous back",
    "a bird with a grey back",
    "a bird with a yellow back",
    "a bird with an olive back",
    "a bird with a green back",
    "a bird with a pink back",
    "a bird with an orange back",
    "a bird with a black back",
    "a bird with a white back",
    "a bird with a red back",
    "a bird with a buff back",
    "a bird with a forked tail",
    "a bird with a rounded tail",
    "a bird with a notched tail",
    "a bird with a fan-shaped tail",
    "a bird with a pointed tail",
    "a bird with a squared tail",
    "a bird with a blue upper tail",
    "a bird with a brown upper tail",
    "a bird with an iridescent upper tail",
    "a bird with a purple upper tail",
    "a bird with a rufous upper tail",
    "a bird with a grey upper tail",
    "a bird with a yellow upper tail",
    "a bird with an olive upper tail",
    "a bird with a green upper tail",
    "a bird with a pink upper tail",
    "a bird with an orange upper tail",
    "a bird with a black upper tail",
    "a bird with a white upper tail",
    "a bird with a red upper tail",
    "a bird with a buff upper tail",
    "a bird with a spotted head",
    "a bird with a malar head",
    "a bird with a crested head",
    "a bird with a masked head",
    "a bird with a unique head",
    "a bird with an eyebrow head",
    "a bird with an eyering head",
    "a bird with a plain head",
    "a bird with an eyeline head",
    "a bird with a striped head",
    "a bird with a capped head",
    "a bird with a blue breast",
    "a bird with a brown breast",
    "a bird with an iridescent breast",
    "a bird with a purple breast",
    "a bird with a rufous breast",
    "a bird with a grey breast",
    "a bird with a yellow breast",
    "a bird with an olive breast",
    "a bird with a green breast",
    "a bird with a pink breast",
    "a bird with an orange breast",
    "a bird with a black breast",
    "a bird with a white breast",
    "a bird with a red breast",
    "a bird with a buff breast",
    "a bird with a blue throat",
    "a bird with a brown throat",
    "a bird with an iridescent throat",
    "a bird with a purple throat",
    "a bird with a rufous throat",
    "a bird with a grey throat",
    "a bird with a yellow throat",
    "a bird with an olive throat",
    "a bird with a green throat",
    "a bird with a pink throat",
    "a bird with an orange throat",
    "a bird with a black throat",
    "a bird with a white throat",
    "a bird with a red throat",
    "a bird with a buff throat",
    "a bird with blue eyes",
    "a bird with brown eyes",
    "a bird with purple eyes",
    "a bird with rufous eyes",
    "a bird with grey eyes",
    "a bird with yellow eyes",
    "a bird with olive eyes",
    "a bird with green eyes",
    "a bird with pink eyes",
    "a bird with orange eyes",
    "a bird with black eyes",
    "a bird with white eyes",
    "a bird with red eyes",
    "a bird with buff eyes",
    "a bird with a bill about the same length as its head",
    "a bird with a bill longer than its head",
    "a bird with a bill shorter than its head",
    "a bird with a blue forehead",
    "a bird with a brown forehead",
    "a bird with an iridescent forehead",
    "a bird with a purple forehead",
    "a bird with a rufous forehead",
    "a bird with a grey forehead",
    "a bird with a yellow forehead",
    "a bird with an olive forehead",
    "a bird with a green forehead",
    "a bird with a pink forehead",
    "a bird with an orange forehead",
    "a bird with a black forehead",
    "a bird with a white forehead",
    "a bird with a red forehead",
    "a bird with a buff forehead",
    "a bird with a blue undertail",
    "a bird with a brown undertail",
    "a bird with an iridescent undertail",
    "a bird with a purple undertail",
    "a bird with a rufous undertail",
    "a bird with a grey undertail",
    "a bird with a yellow undertail",
    "a bird with an olive undertail",
    "a bird with a green undertail",
    "a bird with a pink undertail",
    "a bird with an orange undertail",
    "a bird with a black undertail",
    "a bird with a white undertail",
    "a bird with a red undertail",
    "a bird with a buff undertail",
    "a bird with a blue nape",
    "a bird with a brown nape",
    "a bird with an iridescent nape",
    "a bird with a purple nape",
    "a bird with a rufous nape",
    "a bird with a grey nape",
    "a bird with a yellow nape",
    "a bird with an olive nape",
    "a bird with a green nape",
    "a bird with a pink nape",
    "a bird with an orange nape",
    "a bird with a black nape",
    "a bird with a white nape",
    "a bird with a red nape",
    "a bird with a buff nape",
    "a bird with a blue belly",
    "a bird with a brown belly",
    "a bird with an iridescent belly",
    "a bird with a purple belly",
    "a bird with a rufous belly",
    "a bird with a grey belly",
    "a bird with a yellow belly",
    "a bird with an olive belly",
    "a bird with a green belly",
    "a bird with a pink belly",
    "a bird with an orange belly",
    "a bird with a black belly",
    "a bird with a white belly",
    "a bird with a red belly",
    "a bird with a buff belly",
    "a bird with rounded wings",
    "a bird with pointed wings",
    "a bird with broad wings",
    "a bird with tapered wings",
    "a bird with long wings",
    "a large bird (between 16 to 32 inches)",
    "a small bird (between 5 to 9 inches)",
    "a very large bird (between 32 to 72 inches)",
    "a medium-sized bird (between 9 to 16 inches)",
    "a very small bird (between 3 to 5 inches)",
    "an upright-perching-like waterbird",
    "a chicken-like bird",
    "a long-legged bird",
    "a duck-like bird",
    "an owl-like bird",
    "a gull-like bird",
    "a hummingbird-like bird",
    "a pigeon-like bird",
    "a tree-clinging bird",
    "a hawk-like bird",
    "a sandpiper-like bird",
    "an upland-ground-like bird",
    "a swallow-like bird",
    "a perching-like bird",
    "a bird with a solid back pattern",
    "a bird with a spotted back pattern",
    "a bird with a striped back pattern",
    "a bird with a multi-colored back pattern",
    "a bird with a solid tail pattern",
    "a bird with a spotted tail pattern",
    "a bird with a striped tail pattern",
    "a bird with a multi-colored tail pattern",
    "a bird with a solid belly pattern",
    "a bird with a spotted belly pattern",
    "a bird with a striped belly pattern",
    "a bird with a multi-colored belly pattern",
    "a primarily blue bird",
    "a primarily brown bird",
    "a primarily iridescent bird",
    "a primarily purple bird",
    "a primarily rufous bird",
    "a primarily grey bird",
    "a primarily yellow bird",
    "a primarily olive bird",
    "a primarily green bird",
    "a primarily pink bird",
    "a primarily orange bird",
    "a primarily black bird",
    "a primarily white bird",
    "a primarily red bird",
    "a primarily buff bird",
    "a bird with blue legs",
    "a bird with brown legs",
    "a bird with iridescent legs",
    "a bird with purple legs",
    "a bird with rufous legs",
    "a bird with grey legs",
    "a bird with yellow legs",
    "a bird with olive legs",
    "a bird with green legs",
    "a bird with pink legs",
    "a bird with orange legs",
    "a bird with black legs",
    "a bird with white legs",
    "a bird with red legs",
    "a bird with buff legs",
    "a bird with a blue bill",
    "a bird with a brown bill",
    "a bird with an iridescent bill",
    "a bird with a purple bill",
    "a bird with a rufous bill",
    "a bird with a grey bill",
    "a bird with a yellow bill",
    "a bird with an olive bill",
    "a bird with a green bill",
    "a bird with a pink bill",
    "a bird with an orange bill",
    "a bird with a black bill",
    "a bird with a white bill",
    "a bird with a red bill",
    "a bird with a buff bill",
    "a bird with a blue crown",
    "a bird with a brown crown",
    "a bird with an iridescent crown",
    "a bird with a purple crown",
    "a bird with a rufous crown",
    "a bird with a grey crown",
    "a bird with a yellow crown",
    "a bird with an olive crown",
    "a bird with a green crown",
    "a bird with a pink crown",
    "a bird with an orange crown",
    "a bird with a black crown",
    "a bird with a white crown",
    "a bird with a red crown",
    "a bird with a buff crown",
    "a bird with a solid wing pattern",
    "a bird with a spotted wing pattern",
    "a bird with a striped wing pattern",
    "a bird with a multi-colored wing pattern",
]

# Generate a mapping containing all concept groups in CUB generated
# using a simple prefix tree
CONCEPT_GROUP_MAP = defaultdict(list)
for i, concept_name in enumerate(list(
    np.array(CONCEPT_SEMANTICS)[SELECTED_CONCEPTS]
)):
    group = concept_name[:concept_name.find("::")]
    CONCEPT_GROUP_MAP[group].append(i)


'''
ADDED: remap uncertainty

Definitions from CUB (certainties.txt)
    1 not visible
    2 guessing
    3 probably
    4 definitely

Unc map represents a mapping from the discrete score to a ``mental probability''
'''
DEFAULT_UNC_MAP = [
    {0: 0.5, 1: 0.5, 2: 0.5, 3:0.75, 4:1.0},
    {0: 0.5, 1: 0.5, 2: 0.5, 3:0.75, 4:1.0},
]
def discrete_to_continuous_unc(unc_val, attr_label, unc_map):
    '''
    Yield a continuous prob representing discrete conf val
    Inspired by CBM data processing

    The selected probability should account for whether the concept is on or off
        E.g., if a human is "probably" sure the concept is off
            flip the prob in unc_map
    '''
    unc_val = unc_val.item()
    attr_label = attr_label.item()
    return float(unc_map[int(attr_label)][unc_val])

##########################################################
## ORIGINAL SAMPLER/CLASSES FROM CBM PAPER
##########################################################

class Sampler(object):
    """Base class for all Samplers.
    Every Sampler subclass has to provide an __iter__ method, providing a way
    to iterate over indices of dataset elements, and a __len__ method that
    returns the length of the returned iterators.
    """

    def __init__(self, data_source):
        pass

    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

class StratifiedSampler(Sampler):
    """Stratified Sampling
    Provides equal representation of target classes in each batch
    """
    def __init__(self, class_vector, batch_size):
        """
        Arguments
        ---------
        class_vector : torch tensor
            a vector of class labels
        batch_size : integer
            batch_size
        """
        self.n_splits = int(class_vector.size(0) / batch_size)
        self.class_vector = class_vector

    def gen_sample_array(self):
        try:
            from sklearn.model_selection import StratifiedShuffleSplit
        except:
            print('Need scikit-learn for this functionality')
        import numpy as np

        s = StratifiedShuffleSplit(n_splits=self.n_splits, test_size=0.5)
        X = torch.randn(self.class_vector.size(0),2).numpy()
        y = self.class_vector.numpy()
        s.get_n_splits(X, y)

        train_index, test_index = next(s.split(X, y))
        return np.hstack([train_index, test_index])

    def __iter__(self):
        return iter(self.gen_sample_array())

    def __len__(self):
        return len(self.class_vector)

class CUBDataset(Dataset):
    """
    Returns a compatible Torch Dataset object customized for the CUB dataset
    """

    def __init__(
        self,
        pkl_file_paths,
        use_attr,
        no_img,
        uncertain_label,
        image_dir,
        n_class_attr,
        root_dir='../data/CUB200/',
        path_transform=None,
        transform=None,
        concept_transform=None,
        label_transform=None,
        traveling_birds=False,
        traveling_birds_root_dir=None,
        use_uncertainty_as_competence=False,
        uncertainty_based_random_labels=False,
        unc_map=DEFAULT_UNC_MAP,
    ):
        """
        Arguments:
        pkl_file_paths: list of full path to all the pkl data
        use_attr: whether to load the attributes (e.g. False for simple finetune)
        no_img: whether to load the images (e.g. False for A -> Y model)
        uncertain_label: if True, use 'uncertain_attribute_label' field (i.e. label weighted by uncertainty score, e.g. 1 & 3(probably) -> 0.75)
        image_dir: default = 'images'. Will be append to the parent dir
        n_class_attr: number of classes to predict for each attribute. If 3, then make a separate class for not visible
        transform: whether to apply any special transformation. Default = None, i.e. use standard ImageNet preprocessing
        """
        self.data = []
        self.is_train = any(["train" in path for path in pkl_file_paths])
        if not self.is_train:
            assert any([("test" in path) or ("val" in path) for path in pkl_file_paths])
        for file_path in pkl_file_paths:
            with open(file_path, 'rb') as f:
                self.data.extend(pickle.load(f))
        self.transform = transform
        self.concept_transform = concept_transform
        self.label_transform = label_transform
        self.use_attr = use_attr
        self.no_img = no_img
        self.uncertain_label = uncertain_label
        self.image_dir = image_dir
        self.n_class_attr = n_class_attr
        self.root_dir = root_dir
        self.path_transform = path_transform
        self.traveling_birds = traveling_birds
        self.traveling_birds_root_dir = traveling_birds_root_dir
        self.use_uncertainty_as_competence = use_uncertainty_as_competence
        self.uncertainty_based_random_labels = uncertainty_based_random_labels
        self.unc_map = unc_map
        self.is_val = any(["val" in path for path in pkl_file_paths])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_data = self.data[idx]
        img_path = img_data['img_path']
        if self.path_transform == None:
            img_path = img_path.replace(
                '/juice/scr/scr102/scr/thaonguyen/CUB_supervision/datasets/',
                self.root_dir
            )
            # Trim unnecessary paths
            try:
                if self.traveling_birds:
                    idx = img_path.split('/').index('CUB_200_2011')
                    folder = '/train/' if (self.is_train or self.is_val) else '/test/'
                    img_path = self.traveling_birds_root_dir + folder + '/'.join(img_path.split('/')[idx+2:])
                else:
                    idx = img_path.split('/').index('CUB_200_2011')
                img = None
                for _ in range(5):
                    try:
                        img = Image.open(img_path).convert('RGB')
                        break
                    except:
                        pass
                if img is None:
                    raise ValueError(f"Failed to fetch {img_path} after 5 trials!")
            except:
                img_path_split = img_path.split('/')
                split = 'train' if self.is_train else 'test'
                img_path = '/'.join(img_path_split[:2] + [split] + img_path_split[2:])
                img = Image.open(img_path).convert('RGB')
        else:
            img_path = self.path_transform(img_path)
            img = Image.open(img_path).convert('RGB')

        class_label = img_data['class_label']
        if self.label_transform:
            class_label = self.label_transform(class_label)
        if self.transform:
            img = self.transform(img)

        if self.use_attr:
            if self.uncertain_label:
                attr_label = img_data['uncertain_attribute_label']
            else:
                attr_label = img_data['attribute_label']
            if self.concept_transform is not None:
                attr_label = self.concept_transform(attr_label)
            if self.no_img:
                if self.n_class_attr == 3:
                    one_hot_attr_label = np.zeros(
                        (len(SELECTED_CONCEPTS), self.n_class_attr)
                    )
                    one_hot_attr_label[np.arange(len(SELECTED_CONCEPTS)), attr_label] = 1
                    return one_hot_attr_label, class_label
                else:
                    return attr_label, class_label
            if self.uncertainty_based_random_labels:
                discrete_unc_label = np.array(img_data['attribute_certainty'])[SELECTED_CONCEPTS]
                instance_attr_label = np.array(img_data['attribute_label'])
                competencies = []
                for (discrete_unc_val, hard_concept_val) in zip(discrete_unc_label, instance_attr_label):
                    competencies.append(discrete_to_continuous_unc(discrete_unc_val, hard_concept_val, self.unc_map))
                return img, class_label, torch.FloatTensor(np.random.binomial(1, competencies))
            elif self.use_uncertainty_as_competence:
                discrete_unc_label = np.array(img_data['attribute_certainty'])[SELECTED_CONCEPTS]
                instance_attr_label = np.array(img_data['attribute_label'])
                competencies = []
                for (discrete_unc_val, hard_concept_val) in zip(discrete_unc_label, instance_attr_label):
                    competencies.append(discrete_to_continuous_unc(discrete_unc_val, hard_concept_val, self.unc_map))
                return img, class_label, torch.FloatTensor(attr_label), torch.FloatTensor(np.array(competencies))
            else:
                return img, class_label, torch.FloatTensor(attr_label)
        else:
            return img, class_label


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for
    imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
    """

    def __init__(self, dataset, indices=None):
        # if indices is not provided,
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices

        # if num_samples is not provided,
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices)

        # distribution of classes in the dataset
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1

        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):  # Note: for single attribute dataset
        return dataset.data[idx]['attribute_label'][0]

    def __iter__(self):
        idx = (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))
        return idx

    def __len__(self):
        return self.num_samples


def load_data(
    pkl_paths,
    use_attr,
    no_img,
    batch_size,
    uncertain_label=False,
    n_class_attr=2,
    image_dir='images',
    resampling=False,
    resol=299,
    root_dir='../data/CUB200/',
    num_workers=1,
    concept_transform=None,
    label_transform=None,
    path_transform=None,
    is_chexpert=False,
    traveling_birds_root_dir=None,
    traveling_birds=False,

    additional_sample_transform=None,
    use_uncertainty_as_competence=False,
    uncertainty_based_random_labels=False,
    unc_map=DEFAULT_UNC_MAP,
    augment=True,
):
    """
    Note: Inception needs (299,299,3) images with inputs scaled between -1 and 1
    Loads data with transformations applied, and upsample the minority class if
    there is class imbalance and weighted loss is not used
    NOTE: resampling is customized for first attribute only, so change
    sampler.py if necessary
    """
    resized_resol = int(resol * 256/224)
    is_training = any(['train.pkl' in f for f in pkl_paths])
    additional_sample_transform = (
        (lambda x: x) if additional_sample_transform is None
        else additional_sample_transform
    )

    if is_training and augment:
        if is_chexpert:
            transform = transforms.Compose([
                transforms.CenterCrop((320, 320)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(0.1),
                transforms.ToTensor(),
                additional_sample_transform,
            ])
        else:
            transform = transforms.Compose([
                transforms.ColorJitter(brightness=32/255, saturation=(0.5, 1.5)),
                transforms.RandomResizedCrop(resol),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(), #implicitly divides by 255
                additional_sample_transform,
                transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [2, 2, 2])
            ])
    else:
        if is_chexpert:
            transform = transforms.Compose([
                transforms.CenterCrop((320, 320)),
                transforms.ToTensor(),
                additional_sample_transform,
            ])
        else:
            transform = transforms.Compose([
                transforms.CenterCrop(resol),
                transforms.ToTensor(), #implicitly divides by 255
                additional_sample_transform,
                transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [2, 2, 2])
            ])

    dataset = CUBDataset(
        pkl_file_paths=pkl_paths,
        use_attr=use_attr,
        no_img=no_img,
        uncertain_label=uncertain_label,
        image_dir=image_dir,
        n_class_attr=n_class_attr,
        transform=transform,
        root_dir=root_dir,
        concept_transform=concept_transform,
        label_transform=label_transform,
        path_transform=path_transform,
        traveling_birds_root_dir=traveling_birds_root_dir,
        traveling_birds=traveling_birds,
        use_uncertainty_as_competence=use_uncertainty_as_competence,
        unc_map=unc_map,
        uncertainty_based_random_labels=uncertainty_based_random_labels,
    )
    if is_training:
        drop_last = True
        shuffle = True
    else:
        drop_last = False
        shuffle = False
    if resampling:
        sampler = StratifiedSampler(ImbalancedDatasetSampler(dataset), batch_size=batch_size)
        loader = DataLoader(dataset, batch_sampler=sampler, num_workers=num_workers)
    else:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)
    return loader

def find_class_imbalance(pkl_file, multiple_attr=False, attr_idx=-1):
    """
    Calculate class imbalance ratio for binary attribute labels stored in pkl_file
    If attr_idx >= 0, then only return ratio for the corresponding attribute id
    If multiple_attr is True, then return imbalance ratio separately for each attribute. Else, calculate the overall imbalance across all attributes
    """
    imbalance_ratio = []
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    n = len(data)
    n_attr = len(data[0]['attribute_label'])
    if attr_idx >= 0:
        n_attr = 1
    if multiple_attr:
        n_ones = [0] * n_attr
        total = [n] * n_attr
    else:
        n_ones = [0]
        total = [n * n_attr]
    for d in data:
        labels = d['attribute_label']
        if multiple_attr:
            for i in range(n_attr):
                n_ones[i] += labels[i]
        else:
            if attr_idx >= 0:
                n_ones[0] += labels[attr_idx]
            else:
                n_ones[0] += sum(labels)
    for j in range(len(n_ones)):
        imbalance_ratio.append(total[j]/n_ones[j] - 1)
    if not multiple_attr: #e.g. [9.0] --> [9.0] * 312
        imbalance_ratio *= n_attr
    return imbalance_ratio




##########################################################
## SIMPLIFIED LOADER FUNCTION FOR STANDARDIZATION
##########################################################

def get_concept_descriptions(
    config,
    root_dir=DATASET_DIR,
    seed=42,
    rerun=False,
):
    if root_dir is None:
        root_dir = DATASET_DIR
    seed_everything(seed)

    sampling_percent = config.get("sampling_percent", 1)
    sampling_groups = config.get("sampling_groups", False)

    concept_group_map = CONCEPT_GROUP_MAP.copy()
    n_concepts = len(SELECTED_CONCEPTS)
    selected_concepts = SELECTED_CONCEPTS
    if sampling_percent != 1:
        # Do the subsampling
        if sampling_groups:
            new_n_groups = int(np.ceil(len(concept_group_map) * sampling_percent))
            selected_groups_file = os.path.join(
                root_dir,
                f"selected_groups_sampling_{sampling_percent}.npy",
            )
            if (not rerun) and os.path.exists(selected_groups_file):
                selected_groups = np.load(selected_groups_file)
            else:
                selected_groups = sorted(
                    np.random.permutation(len(concept_group_map))[:new_n_groups]
                )
                np.save(selected_groups_file, selected_groups)
            selected_concepts = []
            group_concepts = [x[1] for x in concept_group_map.items()]
            for group_idx in selected_groups:
                selected_concepts.extend(group_concepts[group_idx])
            selected_concepts = sorted(set(selected_concepts))
        else:
            new_n_concepts = int(np.ceil(n_concepts * sampling_percent))
            selected_concepts_file = os.path.join(
                root_dir,
                f"selected_concepts_sampling_{sampling_percent}.npy",
            )
            if (not rerun) and os.path.exists(selected_concepts_file):
                selected_concepts = np.load(selected_concepts_file)
            else:
                selected_concepts = sorted(
                    np.random.permutation(n_concepts)[:new_n_concepts]
                )
                np.save(selected_concepts_file, selected_concepts)
        # Then we also have to update the concept group map so that
        # selected concepts that were previously in the same concept
        # group are maintained in the same concept group
        new_concept_group = {}
        remap = dict((y, x) for (x, y) in enumerate(selected_concepts))
        selected_concepts_set = set(selected_concepts)
        for selected_concept in selected_concepts:
            for concept_group_name, group_concepts in concept_group_map.items():
                if selected_concept in group_concepts:
                    if concept_group_name in new_concept_group:
                        # Then we have already added this group
                        continue
                    # Then time to add this group!
                    new_concept_group[concept_group_name] = []
                    for other_concept in group_concepts:
                        if other_concept in selected_concepts_set:
                            # Add the remapped version of this concept
                            # into the concept group
                            new_concept_group[concept_group_name].append(
                                remap[other_concept]
                            )
        # And update the concept group map accordingly
        concept_group_map = new_concept_group
    return list(np.array(CONCEPT_SEMANTICS_SENTENCE)[selected_concepts])

def generate_data(
    config,
    root_dir=DATASET_DIR,
    seed=42,
    output_dataset_vars=False,
    rerun=False,
    train_sample_transform=None,
    test_sample_transform=None,
    val_sample_transform=None,
    use_uncertainty_as_competence=False,
    train_augment=True,
    unc_map=DEFAULT_UNC_MAP,
    uncertainty_based_random_labels=False,
):
    if root_dir is None:
        root_dir = DATASET_DIR
    base_dir = os.path.join(root_dir, 'class_attr_data_10')
    seed_everything(seed)
    train_data_path = os.path.join(base_dir, 'train.pkl')
    if config.get('weight_loss', False):
        imbalance = find_class_imbalance(train_data_path, True)
    else:
        imbalance = None

    val_data_path = train_data_path.replace('train.pkl', 'val.pkl')
    test_data_path = train_data_path.replace('train.pkl', 'test.pkl')
    sampling_percent = config.get("sampling_percent", 1)
    sampling_groups = config.get("sampling_groups", False)
    traveling_birds_root_dir = config.get('traveling_birds_root_dir', None)
    traveling_birds = config.get('traveling_birds', False)
    train_augment = config.get('train_augment', train_augment)
    use_uncertainty_as_competence = config.get(
        'use_uncertainty_as_competence',
        use_uncertainty_as_competence,
    )
    uncertainty_based_random_labels = config.get(
        'uncertainty_based_random_labels',
        uncertainty_based_random_labels,
    )
    unc_map = config.get('unc_map', unc_map)

    concept_group_map = CONCEPT_GROUP_MAP.copy()
    n_concepts = len(SELECTED_CONCEPTS)
    if sampling_percent != 1:
        # Do the subsampling
        if sampling_groups:
            new_n_groups = int(np.ceil(len(concept_group_map) * sampling_percent))
            selected_groups_file = os.path.join(
                root_dir,
                f"selected_groups_sampling_{sampling_percent}.npy",
            )
            if (not rerun) and os.path.exists(selected_groups_file):
                selected_groups = np.load(selected_groups_file)
            else:
                selected_groups = sorted(
                    np.random.permutation(len(concept_group_map))[:new_n_groups]
                )
                np.save(selected_groups_file, selected_groups)
            selected_concepts = []
            group_concepts = [x[1] for x in concept_group_map.items()]
            for group_idx in selected_groups:
                selected_concepts.extend(group_concepts[group_idx])
            selected_concepts = sorted(set(selected_concepts))
        else:
            new_n_concepts = int(np.ceil(n_concepts * sampling_percent))
            selected_concepts_file = os.path.join(
                root_dir,
                f"selected_concepts_sampling_{sampling_percent}.npy",
            )
            if (not rerun) and os.path.exists(selected_concepts_file):
                selected_concepts = np.load(selected_concepts_file)
            else:
                selected_concepts = sorted(
                    np.random.permutation(n_concepts)[:new_n_concepts]
                )
                np.save(selected_concepts_file, selected_concepts)
        # Then we also have to update the concept group map so that
        # selected concepts that were previously in the same concept
        # group are maintained in the same concept group
        new_concept_group = {}
        remap = dict((y, x) for (x, y) in enumerate(selected_concepts))
        selected_concepts_set = set(selected_concepts)
        for selected_concept in selected_concepts:
            for concept_group_name, group_concepts in concept_group_map.items():
                if selected_concept in group_concepts:
                    if concept_group_name in new_concept_group:
                        # Then we have already added this group
                        continue
                    # Then time to add this group!
                    new_concept_group[concept_group_name] = []
                    for other_concept in group_concepts:
                        if other_concept in selected_concepts_set:
                            # Add the remapped version of this concept
                            # into the concept group
                            new_concept_group[concept_group_name].append(
                                remap[other_concept]
                            )
        # And update the concept group map accordingly
        concept_group_map = new_concept_group
        print("\t\tSelected concepts:", selected_concepts)
        print(f"\t\tUpdated concept group map (with {len(concept_group_map)} groups):")
        for k, v in concept_group_map.items():
            print(f"\t\t\t{k} -> {v}")

        def concept_transform(sample):
            if isinstance(sample, list):
                sample = np.array(sample)
            return sample[selected_concepts]

        # And correct the weight imbalance
        if config.get('weight_loss', False):
            imbalance = np.array(imbalance)[selected_concepts]
        n_concepts = len(selected_concepts)
    else:
        concept_transform = None


    train_dl = load_data(
        pkl_paths=[train_data_path],
        use_attr=True,
        no_img=False,
        batch_size=config['batch_size'],
        uncertain_label=False,
        n_class_attr=2,
        image_dir='images',
        resampling=False,
        root_dir=root_dir,
        num_workers=config['num_workers'],
        concept_transform=concept_transform,
        traveling_birds_root_dir=traveling_birds_root_dir,
        traveling_birds=traveling_birds,
        additional_sample_transform=train_sample_transform,
        use_uncertainty_as_competence=use_uncertainty_as_competence,
        augment=train_augment,
        unc_map=unc_map,
        uncertainty_based_random_labels=uncertainty_based_random_labels,
    )
    val_dl = load_data(
        pkl_paths=[val_data_path],
        use_attr=True,
        no_img=False,
        batch_size=config['batch_size'],
        uncertain_label=False,
        n_class_attr=2,
        image_dir='images',
        resampling=False,
        root_dir=root_dir,
        num_workers=config['num_workers'],
        concept_transform=concept_transform,
        traveling_birds_root_dir=traveling_birds_root_dir,
        traveling_birds=traveling_birds,
        additional_sample_transform=val_sample_transform,
        use_uncertainty_as_competence=use_uncertainty_as_competence,
        augment=False,
        unc_map=unc_map,
        uncertainty_based_random_labels=uncertainty_based_random_labels,
    )

    test_dl = load_data(
        pkl_paths=[test_data_path],
        use_attr=True,
        no_img=False,
        batch_size=config['batch_size'],
        uncertain_label=False,
        n_class_attr=2,
        image_dir='images',
        resampling=False,
        root_dir=root_dir,
        num_workers=config['num_workers'],
        concept_transform=concept_transform,
        traveling_birds_root_dir=traveling_birds_root_dir,
        traveling_birds=traveling_birds,
        additional_sample_transform=test_sample_transform,
        use_uncertainty_as_competence=use_uncertainty_as_competence,
        augment=False,
        unc_map=unc_map,
        uncertainty_based_random_labels=uncertainty_based_random_labels,
    )
    if not output_dataset_vars:
        return train_dl, val_dl, test_dl, imbalance
    return (
        train_dl,
        val_dl,
        test_dl,
        imbalance,
        (n_concepts, N_CLASSES, concept_group_map),
    )

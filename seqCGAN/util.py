NPRINT_LINE_LEN = 114
# NPRINT_REAL_WIDTH = 50*8
NPRINT_REAL_WIDTH = 22*8
# LABEL_DICT = {'facebook': 0, 'skype': 1, 'email': 2, 'voipbuster': 3, 'youtube': 4, 'ftps': 5, 'vimeo': 6, 'spotify': 7, 'netflix': 8, 'bittorrent': 9}
# LABEL_COUNT_DICT = {'facebook': 0.009872613032664797, 'skype': 0.013418145451480113, 'email': 0.08262515739418122, 'voipbuster': 0.02679248846949511, 'youtube': 0.11559763804444134, 'ftps': 0.19697837522772804, 'vimeo': 0.18104630076077943, 'spotify': 0.17972479491581025, 'netflix': 0.14232541562697112, 'bittorrent': 0.05161907107644865}
LABEL_DICT = {'facebook': 0, 'skype': 1}
LABEL_COUNT_DICT = {'facebook': 0.4238854238854239, 'skype': 0.5761145761145762}
# LABEL_DICT = {'email': 0, 'youtube': 1, 'ftps': 2, 'vimeo': 3, 'spotify': 4, 'netflix': 5, 'bittorrent': 6}
# LABEL_COUNT_DICT = {'email': 0.08698147193341348, 'youtube': 0.12169238796317941, 'ftps': 0.2073638290892577, 'vimeo': 0.19059175467762657, 'spotify': 0.18920057398654902, 'netflix': 0.14982935627836538, 'bittorrent': 0.05434062607160842}
SEQ_DIM = 2
WORD_VEC_SIZE = 8
SEQ_DICT = ['time','pkt_len']
MAX_PKT_LEN = 3001
MAX_TIME = 10000
MAX_PORT = 65535
MAX_SEQ_LEN = 16
N_CRITIC = 5
N_ROLL = 16
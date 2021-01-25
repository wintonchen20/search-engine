from nltk.stem import PorterStemmer
import re
import math
import sklearn

"""
1.master of software engineering : retrieved the top 30 before hand
2.ACM : retrieved the top 30 before hand
3.machine learning : retrieved the top 30 before hand
4.cristina lopes : retrieved the top 30 before hand
5.to be or not to be : the positioning and amount of the same words was messing with my positional index so I added a try except to acccount for the error
6.information retrieval : implremented better cosine similarity 
7.U.S.A : fixed my tokenizer to account for acryonms 
8.USA : Did some tweaks to the tokenizer to make it match more of what U.S.A would return
9.United States of America: Made it so that it matchhed
10: id : included an index of indexes with two letters to improve retrieval 
11. VPN : made everything lowercase to normalize it for searching
12. Irvine: not only lowercased certain letters, but every letter
13. University of California: Had a little more trouble for this so I implemented some stop words to take out of to make indexing faster
14. Alberto : tried to make certain HTML tags weigh more
15. master of software engineering : implemented positional indexing to improve results
16. UCI: Same as VPN, and lowercased everything.
17. Health : I lowercased everything
18. Donald Bren :  Made names matter a bit more
"""

class search_engine():
    def __init__(self):
        

        #This is an in memory dictionary containing the byte positions of the first two letters of words in the file
        # self.index_of_index = { {letter_one]}{letter_two} : byte_position }, example {'ac' : 123421}
        
        self.index_of_index = {'a': 0, 'ab': 838122, 'ac': 1044147, 'ad': 1485958, 'ae': 1972888, 'af': 2001714, 'ag': 2096978, 'ah': 2248614, 'ai': 2260499, 'aj': 2315777, 'ak': 2336140, 
            'al': 2357366, 'am': 3046234, 'an': 3223455, 'ao': 4686913, 'ap': 4697200, 'aq': 6797793, 'ar': 6804819, 'as': 7551203, 'at': 8012350, 'au': 8354564, 'av': 8674117, 'aw': 8758830, 
            'ax': 8844208, 'ay': 8918695, 'az': 8923089, 'b': 8931694, 'ba': 10022915, 'bb': 10508597, 'bc': 10514296, 'bd': 10531786, 'be': 10541972, 'bf': 11452767, 'bg': 11472062, 
            'bh': 11474222, 'bi': 11478847, 'bj': 11763994, 'bk': 11765409, 'bl': 11766515, 'bm': 11988161, 'bn': 12011736, 'bo': 12018589, 'bp': 12363345, 'bq': 12382341, 'br': 12382858, 
            'bs': 12755371, 'bt': 12769097, 'bu': 12776490, 'bv': 13225941, 'bw': 13229701, 'bx': 13235524, 'by': 13237351, 'bz': 13431905, 'c': 13433525, 'ca': 13591725, 'cb': 14815798, 
            'cc': 14900810, 'cd': 14967302, 'ce': 15016240, 'cf': 15161627, 'cg': 15171253, 'ch': 15202564, 'ci': 16013307, 'cj': 16110679, 'ck': 16112148, 'cl': 16116923, 'cm': 16754022, 
            'cn': 16904419, 'co': 16913193, 'cp': 21930817, 'cq': 21953692, 'cr': 21954274, 'cs': 22254050, 'ct': 22341559, 'cu': 22371250, 'cv': 22563155, 'cw': 22573853, 'cx': 22579023, 
            'cy': 22892894, 'cz': 22942314, 'd': 22953316, 'da': 23135407, 'db': 23710562, 'dc': 23774213, 'dd': 23786639, 'de': 23799924, 'df': 25082377, 'dg': 25106855, 'dh': 25110010, 
            'di': 25121374, 'dj': 25823810, 'dk': 25825974, 'dl': 25833901, 'dm': 25847273, 'dn': 25861416, 'do': 25875596, 'dp': 26485853, 'dq': 26491478, 'dr': 26492705, 'ds': 28648144, 
            'dt': 28711239, 'du': 28741348, 'dv': 28873894, 'dw': 28889985, 'dx': 28896590, 'dy': 28898218, 'dz': 28944775, 'e': 28949048, 'ea': 29665295, 'eb': 29976304, 'ec': 29987774, 
            'ed': 30195540, 'ee': 30418354, 'ef': 30491016, 'eg': 30549818, 'eh': 30562191, 'ei': 30577055, 'ej': 30619638, 'ek': 30728260, 'el': 30732904, 'em': 30898089, 'en': 31124947, 
            'eo': 31928831, 'ep': 31979780, 'eq': 32043009, 'er': 32084135, 'es': 32169912, 'et': 32264682, 'eu': 32314534, 'ev': 32349656, 'ew': 32521930, 'ex': 32526863, 'ey': 33357175, 
            'ez': 33369593, 'f': 33374910, 'fa': 33453779, 'fb': 33993401, 'fc': 33999513, 'fd': 34006784, 'fe': 34030268, 'ff': 34322537, 'fg': 34327499, 'fh': 34330659, 'fi': 34333853, 
            'fj': 34841868, 'fk': 34843020, 'fl': 34844270, 'fm': 35063545, 'fn': 35068210, 'fo': 35073936, 'fp': 35831927, 'fq': 35836292, 'fr': 35838144, 'fs': 36366925, 'ft': 36383937, 
            'fu': 36413927, 'fv': 36612511, 'fw': 36613518, 'fx': 36615844, 'fy': 36619674, 'fz': 36621765, 'g': 36624390, 'ga': 36693492, 'gb': 36937656, 'gc': 36942891, 'gd': 36983102, 
            'ge': 36991868, 'gf': 37531925, 'gg': 37555702, 'gh': 37582826, 'gi': 37593778, 'gj': 37769589, 'gk': 37771295, 'gl': 37772193, 'gm': 38898514, 'gn': 38934382, 'go': 38955558, 
            'gp': 39316079, 'gq': 39330429, 'gr': 39331515, 'gs': 39929113, 'gt': 39956961, 'gu': 39977793, 'gv': 40126883, 'gw': 40128358, 'gx': 40165964, 'gy': 40168825, 'gz': 40177833, 
            'h': 40181532, 'ha': 40260997, 'hb': 40886747, 'hc': 40892933, 'hd': 40899122, 'he': 40907532, 'hf': 41349433, 'hg': 41350154, 'hh': 41352780, 'hi': 41354671, 'hj': 41617034, 
            'hk': 41617749, 'hl': 41690677, 'hm': 41696086, 'hn': 41705055, 'ho': 41712255, 'hp': 42098743, 'hq': 42107468, 'hr': 42113708, 'hs': 42129901, 'ht': 42151464, 'hu': 42401534, 
            'hv': 42703382, 'hw': 42704146, 'hx': 42712599, 'hy': 42713030, 'hz': 42793743, 'i': 42794692, 'ia': 43005707, 'ib': 43015595, 'ic': 43027877, 'id': 43152526, 'ie': 43247278, 
            'if': 43285209, 'ig': 43409318, 'ih': 43435604, 'ii': 43442594, 'ij': 43477063, 'ik': 43481656, 'il': 43503342, 'im': 43541391, 'in': 43973002, 'io': 46182035, 'ip': 46218332, 
            'iq': 46238045, 'ir': 46240808, 'is': 46317630, 'it': 46774959, 'iu': 47219829, 'iv': 47220536, 'iw': 47233437, 'ix': 47236757, 'iy': 47242259, 'iz': 47243741, 'j': 47281348, 
            'ja': 47535168, 'jb': 54980345, 'jc': 55415672, 'jd': 55485308, 'je': 55668692, 'jf': 56268349, 'jg': 56277085, 'jh': 56308173, 'ji': 56310872, 'jj': 56382275, 'jk': 56383350, 
            'jl': 56385989, 'jm': 56396941, 'jn': 56510901, 'jo': 56536280, 'jp': 57047078, 'jq': 57132931, 'jr': 57147389, 'js': 57196812, 'jt': 57376209, 'ju': 57418500, 'jv': 57715767, 
            'jw': 57864012, 'jx': 57880290, 'jy': 57887519, 'jz': 57894927, 'k': 57895448, 'ka': 57963553, 'kb': 58148097, 'kc': 58161616, 'kd': 58167036, 'ke': 58176565, 'kf': 58332472, 
            'kg': 58334320, 'kh': 58336890, 'ki': 58346456, 'kj': 58446840, 'kk': 58447755, 'kl': 58448573, 'km': 58478495, 'kn': 58483261, 'ko': 58557256, 'kp': 58646088, 'kq': 58648029, 
            'kr': 58648399, 'ks': 58715247, 'kt': 58726692, 'ku': 58728342, 'kv': 58804344, 'kw': 58806170, 'kx': 58811004, 'ky': 58811713, 'kz': 58818039, 'l': 58818385, 'la': 58970487, 
            'lb': 59401718, 'lc': 59507729, 'ld': 59511836, 'le': 59551581, 'lf': 59972123, 'lg': 59973491, 'lh': 59982518, 'li': 59984398, 'lj': 60741671, 'lk': 60748321, 'll': 60749483, 
            'lm': 60764394, 'ln': 60769714, 'lo': 60773124, 'lp': 61248010, 'lq': 61251590, 'lr': 61252069, 'ls': 61259099, 'lt': 61272358, 'lu': 61277224, 'lv': 61400461, 'lw': 61403682, 
            'lx': 61409648, 'ly': 61410001, 'lz': 61427984, 'm': 61430343, 'ma': 61917848, 'mb': 70534968, 'mc': 70565872, 'md': 70597657, 'me': 70621552, 'mf': 71350705, 'mg': 71383971, 
            'mh': 71414221, 'mi': 71417563, 'mj': 71906277, 'mk': 71916917, 'ml': 71926194, 'mm': 71936881, 'mn': 71970883, 'mo': 71976277, 'mp': 73197825, 'mq': 73208768, 'mr': 73214503, 
            'ms': 73231543, 'mt': 73266051, 'mu': 73288739, 'mv': 73828900, 'mw': 73853134, 'mx': 73855377, 'my': 73861487, 'mz': 74179880, 'n': 74182082, 'na': 74304099, 'nb': 74619560, 
            'nc': 74638698, 'nd': 74650488, 'ne': 74669714, 'nf': 75578736, 'ng': 75589126, 'nh': 75596333, 'ni': 75603638, 'nj': 75738986, 'nk': 75745050, 'nl': 75747297, 'nm': 75752325, 
            'nn': 75787844, 'no': 75793525, 'np': 76415671, 'nq': 76427541, 'nr': 76428405, 'ns': 76691286, 'nt': 76707567, 'nu': 76723667, 'nv': 77019246, 'nw': 77021765, 'nx': 77024190, 
            'ny': 77026060, 'nz': 77069415, 'o': 77078140, 'oa': 77633191, 'ob': 77664362, 'oc': 77865752, 'od': 77978383, 'oe': 78024169, 'of': 78028397, 'og': 78948040, 'oh': 78963816, 
            'oi': 78971493, 'oj': 78975052, 'ok': 78977716, 'ol': 78986761, 'om': 79034121, 'on': 79049801, 'oo': 79448305, 'op': 79465056, 'oq': 79951660, 'or': 79952167, 'os': 85540959, 
            'ot': 85798726, 'ou': 85879267, 'ov': 86395120, 'ow': 86507406, 'ox': 87084734, 'oy': 87094349, 'oz': 87095544, 'p': 87101239, 'pa': 87270220, 'pb': 87973898, 'pc': 87983320, 
            'pd': 88003680, 'pe': 88076393, 'pf': 88589808, 'pg': 88668159, 'ph': 88716425, 'pi': 88861337, 'pj': 89052681, 'pk': 89054593, 'pl': 89063137, 'pm': 90206873, 'pn': 90399384, 
            'po': 90417997, 'pp': 91062318, 'pq': 91116946, 'pr': 91118110, 'ps': 94146641, 'pt': 94241959, 'pu': 94261935, 'pv': 94484534, 'pw': 94494130, 'px': 94500018, 'py': 94510340, 
            'pz': 94545975, 'q': 94546810, 'qa': 94582355, 'qb': 94585755, 'qc': 94586662, 'qd': 94587621, 'qe': 94590165, 'qf': 94594915, 'qg': 94595405, 'qh': 94595764, 'qi': 94596148, 
            'qj': 94607679, 'qk': 94608117, 'ql': 94608733, 'qm': 94609547, 'qn': 94610197, 'qo': 94611197, 'qp': 94615419, 'qq': 94617270, 'qr': 94622056, 'qs': 94622770, 'qt': 94625689, 
            'qu': 94627843, 'qv': 94941954, 'qw': 94942488, 'qx': 94943022, 'qy': 94943359, 'qz': 94943728, 'r': 94944306, 'ra': 95156791, 'rb': 95443476, 'rc': 95464459, 'rd': 95927257, 
            're': 96015382, 'rf': 97914640, 'rg': 97928516, 'rh': 97933459, 'ri': 98009279, 'rj': 98219965, 'rk': 98221752, 'rl': 98223068, 'rm': 98226639, 'rn': 98261345, 'ro': 98274099, 
            'rp': 98513233, 'rq': 98542298, 'rr': 98542853, 'rs': 98548043, 'rt': 98563300, 'ru': 98660639, 'rv': 98860551, 'rw': 98862730, 'rx': 103596601, 'ry': 103598251, 'rz': 103615086, 
            's': 103618844, 'sa': 103889528, 'sb': 104338254, 'sc': 104347242, 'sd': 105048650, 'se': 105071554, 'sf': 106635986, 'sg': 106715921, 'sh': 106737440, 'si': 107187193, 'sj': 107629075, 
            'sk': 107630805, 'sl': 107696833, 'sm': 107881513, 'sn': 107992537, 'so': 108041870, 'sp': 123669915, 'sq': 124375367, 'sr': 124444370, 'ss': 124492411, 'st': 124529849, 'su': 125498933, 
            'sv': 126440802, 'sw': 126487851, 'sx': 126555565, 'sy': 126560165, 'sz': 126801840, 't': 126835160, 'ta': 127011672, 'tb': 127463714, 'tc': 127473704, 'td': 127515279, 'te': 127518753, 
            'tf': 128210047, 'tg': 128214772, 'th': 128259472, 'ti': 130722446, 'tj': 131164584, 'tk': 131167165, 'tl': 131170442, 'tm': 131176439, 'tn': 131197378, 'to': 131203648, 'tp': 132328728, 
            'tq': 132342032, 'tr': 132342432, 'ts': 132975038, 'tt': 132995462, 'tu': 133009562, 'tv': 133193903, 'tw': 133214927, 'tx': 133298872, 'ty': 133318972, 'tz': 133429169, 'u': 133433447, 
            'ua': 133461669, 'ub': 133478882, 'uc': 133502411, 'ud': 133623623, 'ue': 133633734, 'uf': 133635513, 'ug': 133637594, 'uh': 133641419, 'ui': 133645462, 'uj': 133685730, 'uk': 133688680, 
            'ul': 133702284, 'um': 133722999, 'un': 133731904, 'uo': 134260823, 'up': 134262015, 'uq': 134395450, 'ur': 134397877, 'us': 134492352, 'ut': 134762710, 'uu': 135095767, 'uv': 135106599, 
            'uw': 135107984, 'ux': 135109209, 'uy': 135111009, 'uz': 135111673, 'v': 135118668, 'va': 135290135, 'vb': 135762620, 'vc': 135765438, 'vd': 135772719, 've': 135774738, 'vf': 136029963, 
            'vg': 136037461, 'vh': 136045789, 'vi': 136047773, 'vj': 136328334, 'vk': 136328854, 'vl': 136329761, 'vm': 136394724, 'vn': 136405084, 'vo': 136405990, 'vp': 136496449, 'vq': 136500671, 
            'vr': 136501104, 'vs': 136505201, 'vt': 136517126, 'vu': 136520276, 'vv': 136525746, 'vw': 136526819, 'vx': 136527738, 'vy': 136528237, 'vz': 136530383, 'w': 136531409, 'wa': 136571789, 
            'wb': 136913207, 'wc': 136914932, 'wd': 136919534, 'we': 136925276, 'wf': 137719237, 'wg': 137721214, 'wh': 137726510, 'wi': 138039674, 'wj': 138756159, 'wk': 138756732, 'wl': 138757377, 
            'wm': 138760009, 'wn': 138764264, 'wo': 138767063, 'wp': 139038323, 'wq': 139045353, 'wr': 139046563, 'ws': 139137993, 'wt': 139261818, 'wu': 139297692, 'wv': 139306296, 'ww': 139306821, 
            'wx': 139358709, 'wy': 139359255, 'wz': 139365917, 'x': 139366480, 'xa': 139579921, 'xb': 139585782, 'xc': 139612433, 'xd': 139627284, 'xe': 139657281, 'xf': 139676570, 'xg': 139709469, 
            'xh': 139709994, 'xi': 139713246, 'xj': 139725264, 'xk': 139740100, 'xl': 139741337, 'xm': 139748607, 'xn': 139928222, 'xo': 139933580, 'xp': 139935473, 'xq': 139942104, 'xr': 139944818, 
            'xs': 139948976, 'xt': 139975897, 'xu': 139980307, 'xv': 139984566, 'xw': 139987068, 'xx': 139993986, 'xy': 140001337, 'xz': 140005688, 'y': 140006148, 'ya': 140047967, 'yb': 140088837, 
            'yc': 140089894, 'yd': 140090945, 'ye': 140091632, 'yf': 140171622, 'yg': 140172544, 'yh': 140172989, 'yi': 140174043, 'yj': 140189494, 'yk': 140190322, 'yl': 140191111, 'ym': 140192033, 
            'yn': 140196495, 'yo': 140197707, 'yp': 140432625, 'yq': 140433977, 'yr': 140434504, 'ys': 140435554, 'yt': 140436904, 'yu': 140439389, 'yv': 140458245, 'yw': 140459308, 'yx': 140462443, 
            'yy': 140462986, 'yz': 140464527, 'z': 140465255, 'za': 140499251, 'zb': 140527887, 'zc': 140532392, 'zd': 140535699, 'ze': 140539364, 'zf': 140575858, 'zg': 140585298, 'zh': 140586351, 
            'zi': 140599450, 'zj': 140624738, 'zk': 140625340, 'zl': 140632375, 'zm': 140635582, 'zn': 140638593, 'zo': 140655524, 'zp': 140685347, 'zq': 140686359, 'zr': 140686701, 'zs': 140687593, 
            'zt': 140689349, 'zu': 140691176, 'zv': 140705757, 'zw': 140709483, 'zx': 140713639, 'zy': 140715666, 'zz': 140724195
            }

        #I normalized each line in the file of docIDs to urls to the longest string which was of length 233.
        #If we were to reach the line containing a docID, we would only have to multiply the docID by length 233 to reach its byte position
        #The file is separated by docID = url | number of words ?(arbitrary amount of ?s)
        self.docIDS_to_urls_num = 233
        
        self.premade_indexes = {
            'master of software engineering':['https://www.informatics.uci.edu/very-top-footer-menu-items/news/page/30/', 'https://mswe.ics.uci.edu/career/interviewing/', 'https://mswe.ics.uci.edu/admissions/information_sessions/', 'https://isg.ics.uci.edu/faculty2/avinash-kumar/', 'https://isg.ics.uci.edu/faculty2/harandizadeh-bahareh/', 'https://mswe.ics.uci.edu/contact-us/', 'https://mswe.ics.uci.edu/program/', 'https://mswe.ics.uci.edu/admissions/', 'https://mswe.ics.uci.edu/people/', 'https://mswe.ics.uci.edu/people/faculty-staff/', 'https://mswe.ics.uci.edu/career/mswe-career-services-2/', 'https://mswe.ics.uci.edu/admissions/cost-and-financial-aid/', 'https://mswe.ics.uci.edu/career/', 'https://mswe.ics.uci.edu/career/additional-resources/', 'https://mswe.ics.uci.edu/career/resources-for-international-students/', 'https://www.informatics.uci.edu/uci-ranked-3rd-best-value-ux-design-graduate-program/#content', 'https://www.stat.uci.edu/2017-ics-deans-award-winners/#more-468', 'https://www.informatics.uci.edu/2015/03/#content', 'https://www.ics.uci.edu/~rickl/courses/cs-171/2015-wq-cs171/CS-171-WQ-2015.htm', 'https://www.cs.uci.edu/art-and-computers-symposium/acsymposium-speakers/#elementor-action%3Aaction%3Dpopup%3Aclose%20settings%3DeyJkb19ub3Rfc2hvd19hZ2FpbiI6IiJ9', 'https://mswe.ics.uci.edu/admissions/admissions-overview/', 'https://www.informatics.uci.edu/grad/mhcid/#content', 'https://www.informatics.uci.edu/from-homebound-to-school-bound-with-telepresence-robots/#content', 'https://www.informatics.uci.edu/seeing-video-games-in-a-new-light/#content', 'https://www.informatics.uci.edu/connected-learning-through-minecraft/', 'https://www.informatics.uci.edu/overseeing-your-online-afterlife/#content', 'https://www.informatics.uci.edu/2018/07/#content', 'https://www.informatics.uci.edu/autism-appjam-highlights-academias-growing-impact-on-the-autism-community/#content', 'http://hai.ics.uci.edu/people.html', 'https://www.informatics.uci.edu/undergrad/student-profiles/diane-jiea-monchusap/#content'],
            'cristina lopes':['https://www.ics.uci.edu/faculty/faculty_dept?department=Informatics', 'https://www.ics.uci.edu/faculty/?department=Informatics', 'https://hombao.ics.uci.edu/?s=news', 'https://hombao.ics.uci.edu/?s=people', 'https://www.ics.uci.edu/community/news/view_news.php?id=670', 'https://www.ics.uci.edu/community/news/view_news.php?id=1084', 'https://www.ics.uci.edu/community/news/articles/view_article?id=73', 'http://flamingo.ics.uci.edu/releases/3.0/src/filtertree/data/female_names.txt', 'https://www.cs.uci.edu/art-and-computers-symposium/acsymposium-speakers/#elementor-action%3Aaction%3Dpopup%3Aclose%20settings%3DeyJkb19ub3Rfc2hvd19hZ2FpbiI6IiJ9', 'https://www.ics.uci.edu/~dechter/courses/ics-275a/spring-2014/project.shtml', 'https://www.ics.uci.edu/~kay/femalenames.txt', 'http://archive.ics.uci.edu/ml/datasets/Tic-Tac-Toe+Endgame', 'http://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Original)', 'http://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Prognostic)', 'http://archive.ics.uci.edu/ml/datasets/Dermatology', 'http://archive.ics.uci.edu/ml/support/Dermatology#692880d7b3356df64bfa0f06a683f89e4ce6955b', 'http://archive.ics.uci.edu/ml/datasets/Robot+Execution+Failures', 'http://archive.ics.uci.edu/ml/datasets/Breast+Cancer', 'https://mailman.ics.uci.edu/mailman/listinfo/ivecg-community', 'https://mailman.ics.uci.edu/mailman/subscribe/ivecg-faculty', 'https://mailman.ics.uci.edu/mailman/options/ivecg-community', 'https://mailman.ics.uci.edu/mailman/listinfo/ivecg-faculty', 'https://mailman.ics.uci.edu/mailman/subscribe/ivecg-community', 'https://mswe.ics.uci.edu/people/faculty-staff/', 'https://redmiles.ics.uci.edu/publication/', 'http://sdcl.ics.uci.edu/papers/', 'http://sdcl.ics.uci.edu/2016/03/congratulations-dr-gerald/', 'https://www.cs.uci.edu/ics-staff-faculty-honored-at-inaugural-faculty-staff-awards-celebration/', 'https://www.cs.uci.edu/multidepartmental-collaboration-on-detecting-code-clones-leads-to-distinguished-paper-award/', 'https://www.ics.uci.edu/~givargis/'],
            'machine learning':['https://cbcl.ics.uci.edu/doku.php/http/www.stanford.edu/boyd/start?idx=software', 'https://cbcl.ics.uci.edu/doku.php/start?rev=1396651880', 'https://cbcl.ics.uci.edu/doku.php/people?rev=1392156622', 'https://cbcl.ics.uci.edu/doku.php/publications?rev=1534399600', 'https://cbcl.ics.uci.edu/doku.php/publications?rev=1516475524', 'https://cbcl.ics.uci.edu/doku.php/start?idx=software', 'https://cbcl.ics.uci.edu/doku.php/publications?rev=1507871734', 'https://cbcl.ics.uci.edu/doku.php/people?rev=1418014452', 'https://cbcl.ics.uci.edu/doku.php/teaching/cs295w11/start?rev=1490825918', 'https://cbcl.ics.uci.edu/doku.php/teaching/cs295w11/start?rev=1490827313', 'https://cbcl.ics.uci.edu/doku.php/software?rev=1397844231', 'https://cbcl.ics.uci.edu/doku.php/data?rev=1407727888', 'https://cbcl.ics.uci.edu/doku.php/teaching?rev=1427745621', 'https://cbcl.ics.uci.edu/doku.php/contact?rev=1360894708', 'https://cbcl.ics.uci.edu/doku.php?id=http:www.ics.uci.edu:xhx:courses:convexopt:projects:approximation.pdf', 'https://cbcl.ics.uci.edu/doku.php/people?rev=1418330896', 'https://cbcl.ics.uci.edu/doku.php/software?rev=1360950402', 'https://cbcl.ics.uci.edu/doku.php/data?rev=1361336451', 'https://cbcl.ics.uci.edu/doku.php/data?rev=1407727068', 'https://cbcl.ics.uci.edu/doku.php/teaching/cs285s14/start?rev=1490826928', 'https://cbcl.ics.uci.edu/doku.php/teaching/cs295w11/start?rev=1490812973', 'https://cbcl.ics.uci.edu/doku.php/software?rev=1360950625', 'https://cbcl.ics.uci.edu/doku.php/software/sgd?rev=1361581267', 'https://cbcl.ics.uci.edu/doku.php/publications?rev=1534400717', 'https://cbcl.ics.uci.edu/doku.php/software?rev=1397843845', 'https://cbcl.ics.uci.edu/doku.php?id=teaching:cs285s14:start', 'https://cbcl.ics.uci.edu/doku.php/data?rev=1407726237', 'https://cbcl.ics.uci.edu/doku.php/http/www.ics.uci.edu/xhx/courses/convexopt/start?idx=software', 'https://cbcl.ics.uci.edu/doku.php?id=http:www.ics.uci.edu:xhx:courses:convexopt:optimality_conditions.pdf', 'https://cbcl.ics.uci.edu/doku.php/teaching/cs285s14/math_227c_scribe_notes_signup?idx=software'],
            'ACM':['https://cbcl.ics.uci.edu/doku.php/publications?rev=1516475524', 'https://cbcl.ics.uci.edu/doku.php/publications?rev=1507871734', 'https://cbcl.ics.uci.edu/doku.php/publications?rev=1534400717', 'https://cml.ics.uci.edu/tag/news/page/5/?page=events&subPage=dss_schedule', 'https://cml.ics.uci.edu/category/news/page/3/?page=people&subPage=faculty', 'https://cml.ics.uci.edu/category/news/?page=events&subPage=dss_schedule', 'https://duttgroup.ics.uci.edu/publications/?limit=3#tppubs', 'https://duttgroup.ics.uci.edu/publications/?limit=6#content', 'https://duttgroup.ics.uci.edu/publications/?limit=4', 'https://duttgroup.ics.uci.edu/publications/?limit=7#tppubs', 'https://duttgroup.ics.uci.edu/publications/?limit=5', 'https://duttgroup.ics.uci.edu/publications/?limit=9#tppubs', 'https://duttgroup.ics.uci.edu/publications/?limit=2', 'https://duttgroup.ics.uci.edu/publications/?limit=8', 'https://grape.ics.uci.edu/wiki/public/wiki/cs222p-2018-fall?version=31', 'https://grape.ics.uci.edu/wiki/public/wiki/cs222-2018-fall?version=39&format=txt', 'https://grape.ics.uci.edu/wiki/public/wiki/cs222p-2017-fall?version=35&format=txt', 'https://grape.ics.uci.edu/wiki/public/wiki/cs222p-2017-fall?version=82&format=txt', 'https://grape.ics.uci.edu/wiki/public/wiki/cs222-2019-fall?version=5&format=txt', 'https://grape.ics.uci.edu/wiki/public/wiki/cs222-2018-fall?format=txt', 'https://grape.ics.uci.edu/wiki/public/wiki/cs222-2017-fall?action=diff&version=26', 'https://isg.ics.uci.edu/events/?post_type=tribe_events&eventDisplay=default', 'https://isg.ics.uci.edu/publications/?limit=3', 'https://isg.ics.uci.edu/event/large-scale-and-low-latency-data-distribution-from-database-to-servers/?ical=1', 'https://isg.ics.uci.edu/publications/?limit=4', 'https://isg.ics.uci.edu/event/pat-helland-theres-no-substitute-for-interchangeability/?ical=1', 'https://isg.ics.uci.edu/event/speaker-david-lomet-microsoft-research-cost-performance-in-modern-data-stores-how-data-caching-systems-succeed/?ical=1', 'https://isg.ics.uci.edu/events/list/?tribe_event_display=past&tribe_paged=1', 'https://isg.ics.uci.edu/event/prof-jeff-ullman-visit/?ical=1', 'https://jgarcia.ics.uci.edu/?rest_route=']
            }
            


        self.punc_list =  {'!': ' ', '@': ' ', '#': ' ', '$': ' ', '©': ' ', '%': ' ', '^': ' ', '&': ' ', '*': ' ', '(': ' ', ')': ' ', '_': ' ', '-': ' ', '=': ' ', 
                            '+': ' ', '"': ' ', ':': ' ', ';': ' ', '<': ' ', '>': ' ', ',': ' ', '.': ' ', '?': ' ', '/': ' ', '{': ' ', '}': ' ', '[': ' ', ']': ' ', 
                            '`': ' ', '~': ' ', '\\': ' ', '|': ' ', '™': ' ', '•': ' ','—':' ','–':' ', "“":" ", "”":" ","‹":" ","›":" ","‘":" ","³":" ","0":" ","1":" ",
                            "2":" ","3":" ","4":" ","5":" ","6":" ","7":" ","8":" ","9":" "
                            }

        self.regexStr = re.compile("[^a-zA-Z0-9]")   #O(1)

        self.ps = PorterStemmer() #Porter Stemming method

        self.common_stop_words = {'few', 'that', 'can', 'doe', 'no', 'off', 'am', 'herself', 'under', 'is', 'hi', 'just', 'but', 'yourself', 'have', 'befor', 'myself', 
            'all', 'wa', 'don', 'when', 'on', 'of', 'themselv', 'by', 'in', 'same', 'who', 'into', 's', 'been', 'until', 'onli', 'their', 'are', 'through', 'onc', 'than', 
            'below', 'over', 'with', 'so', 'your', 'an', 'too', 'ani', 'me', 'her', 'veri', 'abov', 'becaus', 'and', 'i', 'you', 'what', 'against', 'both', 'for', 'ourselv', 
            'there', 'at', 'after', 'where', 'then', 'some', 'itself', 'him', 'other', 'ha', 'yourselv', 'further', 'dure', 'a', 'while', 'he', 'out', 'they', 'if', 'up', 'these', 
            'nor', 'should', 'the', 'each', 'about', 'our', 'them', 'she', 'how', 'more', 'between', 'do', 'whom', 'most', 'down', 'whi', 'again', 'himself', 'had', 'from', 'we', 
            'own', 'those', 'as', 'it', 'now', 'such', 'thi', 'did', 'my', 'will', 'were', 'which', 'here'
            } #Copied from https://gist.github.com/sebleier/554280
        
        self.total_documents = 14071 #The total number of documents in the corpus

        self.docID_to_urls = {} #A dictionary holding key-value pairs of docIDs:url

        self.docID_to_number_of_tokens = {} #A dictionary holding key-value pairs of docIDs:amount of tokens in it

    def tokenize(self,words:str):
        """
        Returns a list of tokens 
        Partially found and updated from https://towardsdatascience.com/benchmarking-python-nlp-tokenizers-3ac4735100c5
        """
        words = words.lower()

        words = words.replace("\n", " ").replace("\r", " ")
        
        t = str.maketrans(self.punc_list)

        words = words.translate(t)

        t = str.maketrans(dict.fromkeys("'`",""))
        words = words.translate(t)

        tokens = re.split(self.regexStr,words)

        #Stems the token using the Porter Stem method
        tokens = [self.ps.stem(token) for token in tokens]
        
        return tokens
    
    def setup(self):
        """
        Opens up a file to access the docIDs, their urls and number of tokens
        """
        with open('docID_word_count.txt', 'r') as readfile:
            for line in readfile:
                docID = line[:line.find('=')]
                url = line[line.find('=')+1:line.find('|')]
                number_of_tokens = line[line.find('|')+1:line.find('?')]

                self.docID_to_urls[docID] = url
                if number_of_tokens == '':
                    self.docID_to_number_of_tokens[docID] = 0
                else:
                    self.docID_to_number_of_tokens[docID] = int(number_of_tokens)
    
    def search(self):
        """
        Given a term, we will find a way to compute the top five search results.
        """
        
        #Sets up some stuff to increase performance
        self.setup()

        try:
            while True:

                #A dictionary holding the document and whether or not the query positional indexes match that of the document and term positonal indexes placement
                #positional_index = { docID : True or False }
                positional_index = {}
                #A dictionary to hold the term and their positions in the query
                term_positions = {}   

                # A term dictionary to hold the terms and their postings. 
                # term_dict = { term : { docID : [positional_indexes of term ] } }
                term_dict = {}

                #An iterator int object to iterate through all of the terms 
                term_iterator = 0

                #A dictionary hold the term and the doc frequency
                term_doc_freq = {}

                #A dictionary holding the term and the tf_idf score
                term_tfidf = {} #term_tfidf = { term : tfidf }

                #A dictionary holding the document and its tfidf score to that term
                doc_tfidf = {} #doc_tfidf = { docID: { term: doc_tfidf } }

                #A to be checked positional index for the first one
                checked_position = set()

                #Query consine normlization
                term_normalization = 0

                #Document cosine normliztion 
                doc_normalization = 0   

                #positonal index to hold the amount of urls and see whether or not it is true 
                positonal_index = set()

                #Ask user for input
                user_input = input("What would you like to search(type 'q' to quit): ") 

                if user_input in ['master of software engineering',"ACM","cristina lopes","machine learning"]:
                    for i in self.premade_indexes[user_input]:
                        print(i)
                    user_input = input("What would you like to search: ") 
                
                if user_input == 'q':
                    break

                #Tokenizes the user_input term 
                term_tokens = self.tokenize(user_input)

                user_input = user_input.split()

                #Adds to the term positions
                if len(term_tokens) != 1:
                    for term in range(len(term_tokens)):
                        term_positions[term_tokens[term]] = term

                #We sort the terms so that is much quicker to search through all the terms
                term_tokens = sorted(term_tokens)

                #If the term is greater than three, remove the some common stop words to improve the query and response time. 
                if len(term_tokens) >= 3:
                    for term in term_tokens.copy():
                        if term in self.common_stop_words:
                            if len(term_tokens) == 2:
                                break
                            else:
                                term_tokens.remove(term)

                with open('full_merged_index.txt','r') as readfile:
                    
                    #Seeks the position of where the first term should be
                    if len(term_tokens[term_iterator]) > 1:  
                        readfile.seek(self.index_of_index[term_tokens[term_iterator][0:2]])
                    else:
                        readfile.seek(self.index_of_index[term_tokens[term_iterator][0]])

                    #While we did not find all of the postings yet
                    while True:
                        #Line equals the file readline
                        line = readfile.readline()

                        token_length = len(term_tokens[term_iterator])

                        #If the token of the readline equals that of the term
                        if term_tokens[term_iterator] == line[:token_length]:
                            
                            line = line[:-2] #removes the newline character and some other mumbo jumbo before it

                            #Adds the term with an empty dictionary inside of term_dict
                            term_dict[term_tokens[term_iterator]] = {}

                            #Adds the term and its frequency to the dict
                            term_doc_freq[term_tokens[term_iterator]] = line[line.find('|')+1:line.find(':')]

                            #value_string holds the docID:[positional_index]
                            value_string = line[line.find(':')+1:]
                            
                            #Splits the value string by each posting and puts it into a list
                            postings = value_string.split('|')

                            #term tf score for query 
                            tf = 1 + math.log(term_tokens.count(term_tokens[term_iterator])/len(term_tokens))
                            
                            #Iterates through every item in the postings list
                            for item in postings:

                                #This is the front token
                                posting_doc_id = item[:item.find(':')]

                                #These are the postings
                                end_list = item[item.find(':')+2:-1].split(',')

                                replace_list = []
                                for posting in range(len(end_list)):
                                    if posting == 0:
                                        replace_list.append(int(end_list[posting]))
                                    else:
                                        replace_list.append(int(end_list[posting][1:]))
                                
                                try:
                                    if term_iterator == 0:
                                        term_dict[term_tokens[0]][posting_doc_id] = replace_list
                                        positional_index[posting_doc_id] = True
                                        checked_position.add(posting_doc_id)
                                    else:
                                        if posting_doc_id not in positional_index:
                                            pass
                                        else:
                                            to_be_compared_list = term_dict[term_tokens[0]][posting_doc_id]
                                            position_of_term_0 = term_positions[term_tokens[0]]
                                            
                                            if posting_doc_id in checked_position:
                                                checked_position.remove(posting_doc_id)

                                            is_the_position_right = False
                                            if position_of_term_0 > term_positions[term_tokens[term_iterator]]:
                                                for position in to_be_compared_list:
                                                    if (position - (position_of_term_0 - term_positions[term_tokens[term_iterator]])) in replace_list:
                                                        is_the_position_right = True
                                                        break
                                            else:
                                                for position in to_be_compared_list:
                                                    if (position + ( term_positions[term_tokens[term_iterator]] - position_of_term_0)) in replace_list:
                                                        is_the_position_right = True
                                                        break
                                            if is_the_position_right == False:
                                                del positional_index[posting_doc_id]
                                except:
                                    pass
                                    

                                #idf score of the term
                                idf = math.log(self.total_documents/(len(postings)))

                                #Adds the query term tf-idf score
                                #term_tfidf = { term : tfidf }
                                query = tf * idf
                                term_tfidf[term_tokens[term_iterator]] = query 
                                
                                #Query normalization
                                term_normalization += (query**2)
                                
                                #document tf score
                                if self.docID_to_number_of_tokens[posting_doc_id] != 0:
                                    doc_tf = 1 + math.log(len(replace_list)/self.docID_to_number_of_tokens[posting_doc_id])
                                else:
                                    doc_tf = 1

                                #Document normalization
                                doc_cosine = doc_tf*idf

                                #Adds the document id with term and document tf-idf score
                                #doc_tfidf = { docID: { term: doc_tfidf } }
                                if posting_doc_id not in doc_tfidf:
                                    doc_tfidf[posting_doc_id] = {term_tokens[term_iterator]:doc_cosine}
                                else:
                                    doc_tfidf[posting_doc_id][term_tokens[term_iterator]] = doc_cosine
                                
                                doc_normalization += (doc_cosine**2)
                            
                            try:
                                if term_iterator > 0:
                                    for id in checked_position:
                                        if id in positional_index:
                                            del positional_index[id]
                                    checked_position.clear()
                                    for key in positional_index:
                                        checked_position.add(key)
                            except:
                                pass

                            term_iterator += 1
                            
                            if term_iterator == len(term_tokens):
                                break
                
                #1/sqrt(sum of weights)
                term_normalization = 1/math.sqrt(term_normalization)
                doc_normalization = 1/math.sqrt(doc_normalization)

                """
                #A dictionary holding the term and the tf_idf score
                term_tfidf = {} #term_tfidf = { term : tfidf }

                #A dictionary holding the document and its tfidf score to that term
                doc_tfidf = {} #doc_tfidf = { docID: { term: doc_tfidf } }
                """

                #This is the portion where you compute consine-similarity and positional_indexing weight
                """
                Cosine-similarity: 
                """
                #Normalization
                term_tfidf_list = [[]]
                for index in term_tfidf:
                    term_tfidf_list[0].append(term_tfidf[index]*term_normalization)
                
                amount_of_terms = len(term_tfidf_list[0])

                doc_cosine_similarity = {}

                i = 0
                doc_tfidf_list = []
                for doc in doc_tfidf:
                    j = 0
                    doc_cosine_similarity[doc] = 0
                    for term in doc_tfidf[doc]:
                        if j == 0:
                            if doc in positonal_index: #If the positions match up with the query, add 0.25 weight
                                doc_tfidf_list.append([doc_tfidf[doc][term]*doc_normalization+0.25])
                            else:  
                                doc_tfidf_list.append([doc_tfidf[doc][term]*doc_normalization])
                        else:
                            if doc in positonal_index: #If the positions match up with the query, add 0.25 weight
                                doc_tfidf_list[i].append(doc_tfidf[doc][term]*doc_normalization+0.25)
                            else:
                                doc_tfidf_list[i].append(doc_tfidf[doc][term]*doc_normalization)
                        j += 1
                    if j != amount_of_terms:
                        doc_tfidf_list[i] += ((amount_of_terms - j) * [0])
                    i += 1

                
                x = sklearn.metrics.pairwise.cosine_similarity(term_tfidf_list, doc_tfidf_list)
                x_index = 0
                for doc in doc_cosine_similarity:
                    doc_cosine_similarity[doc] = x[0][x_index]
                    x_index += 1
                
                doc_cosine_similarity = sorted(doc_cosine_similarity, key=lambda x: doc_cosine_similarity[x], reverse = True)
                
                print("retrieving the top 30 urls")
                for url in doc_cosine_similarity[0:30]:
                    print(self.docID_to_urls[url])
        except:
            print("There was an error, that might be due to bad spelling leading to a word that was not in the data base.")
            print("Please try again")

    def tfidf(self):
        """
        Creates a tf-idf score per term per document using the formula:
            tf-idf = ( 1 + log(tf)) * log(N/df)
            tf = (number of times term t appears in a document)/ (total number of terms in the document)
            N/df = (total number of documents) / (number of documents with term t in it)
        """
        #for token in self.docID_hashmap:
            #for document in self.docID_hashmap[token]:
            #Now it implememnts a two element list associated with each token: docID, frequency of that token, and tf-idf score
                #word_freq = self.docID_hashmap[token][document][0]
                #tf = word_freq / self.docID_word_count[document][1]
                #idf = self.docID / len(self.docID_hashmap[token])
                #self.tf_idf_dict[document] = (1 + math.log(tf)) * math.log(idf)
        pass
                        

if __name__ == '__main__':
    whatever = search_engine()
    whatever.search()
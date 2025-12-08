# Распознание контекста фразы

Определение контекста фразы как позитивной или негативной

```python
import re
from navec import Navec
from tqdm import tqdm
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim


class PhraseDataset(data.Dataset):
    def __init__(self, path_true, path_false, navec_emb, batch_size=8):
        self.navec_emb = navec_emb
        self.batch_size = batch_size

        with open(path_true, 'r', encoding='utf-8') as f:
            phrase_true = f.readlines()
            self._clear_phrase(phrase_true)

        with open(path_false, 'r', encoding='utf-8') as f:
            phrase_false = f.readlines()
            self._clear_phrase(phrase_false)

        self.phrase_lst = [(_x, 0) for _x in phrase_true] + [(_x, 1) for _x in phrase_false]
        self.phrase_lst.sort(key=lambda _x: len(_x[0]))
        self.dataset_len = len(self.phrase_lst)

    def _clear_phrase(self, p_lst):
        for _i, _p in enumerate(p_lst):
            _p = _p.lower().replace('\ufeff', '').strip()
            _p = re.sub(r'[^А-яA-z- ]', '', _p)
            _words = _p.split()
            _words = [w for w in _words if w in self.navec_emb]
            p_lst[_i] = _words

    def __getitem__(self, item):
        item *= self.batch_size
        item_last = item + self.batch_size
        if item_last > self.dataset_len:
            item_last = self.dataset_len

        _data = []
        _target = []
        max_length = len(self.phrase_lst[item_last - 1][0])

        for i in range(item, item_last):
            words_emb = []
            phrase = self.phrase_lst[i]
            length = len(phrase[0])

            for k in range(max_length):
                t = torch.tensor(self.navec_emb[phrase[0][k]], dtype=torch.float32) if k < length else torch.zeros(300)
                words_emb.append(t)

            _data.append(torch.vstack(words_emb))
            _target.append(torch.tensor(phrase[1], dtype=torch.float32))

        _data_batch = torch.stack(_data)
        _target = torch.vstack(_target)
        return _data_batch, _target

    def __len__(self):
        last = 0 if self.dataset_len % self.batch_size == 0 else 1
        return self.dataset_len // self.batch_size + last


class WordsRNN(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.hidden_size = 16
        self.in_features = in_features
        self.out_features = out_features

        self.rnn = nn.RNN(in_features, self.hidden_size, batch_first=True, bidirectional=True)
        self.out = nn.Linear(self.hidden_size * 2, out_features)

    def forward(self, x):
        x, h = self.rnn(x)
        hh = torch.cat((h[-2, :, :], h[-1, :, :]), dim=1)
        y = self.out(hh)
        return y


path = 'navec_hudlit_v1_12B_500K_300d_100q.tar'
navec = Navec.load(path)

d_train = PhraseDataset("train_data_true", "train_data_false", navec)
train_data = data.DataLoader(d_train, batch_size=1, shuffle=True)

model = WordsRNN(300, 1)

optimizer = optim.Adam(params=model.parameters(), lr=0.001, weight_decay=0.001)
loss_func = nn.BCEWithLogitsLoss()

epochs = 20
model.train()

for _e in range(epochs):
    loss_mean = 0
    lm_count = 0

    train_tqdm = tqdm(train_data, leave=True)
    for x_train, y_train in train_tqdm:
        predict = model(x_train.squeeze(0)).squeeze(0)
        loss = loss_func(predict, y_train.squeeze(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lm_count += 1
        loss_mean = 1 / lm_count * loss.item() + (1 - 1 / lm_count) * loss_mean
        train_tqdm.set_description(f"Epoch [{_e + 1}/{epochs}], loss_mean={loss_mean:.3f}")

st = model.state_dict()
torch.save(st, 'model_rnn_bidir.tar')

model.eval()

phrase = "Сегодня пасмурная погода"
phrase_lst = phrase.lower().split()
phrase_lst = [torch.tensor(navec[w]) for w in phrase_lst if w in navec]
_data_batch = torch.stack(phrase_lst)
predict = model(_data_batch.unsqueeze(0)).squeeze(0)
p = torch.nn.functional.sigmoid(predict).item()
print(p)
print(phrase, ":", "положительное" if p < 0.5 else "отрицательное")
```

Распознавания паронимов

```python
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim

# @title Глобальные переменные
_global_words_0 = ['аа', 'аатаа', 'аба', 'абба', 'абиба', 'ава', 'аваава', 'авава', 'авва', 'ага', 'агга', 'ада',
                   'адда', 'ажа', 'ака', 'акака', 'аканака', 'акика', 'акка', 'ала', 'алала', 'алафала', 'амакама',
                   'амма', 'ана', 'анапана', 'анасана', 'анатана', 'анисина', 'анна', 'анона', 'апа', 'апипа', 'аппа',
                   'ара', 'арара', 'арора', 'арра', 'арура', 'аса', 'ата', 'атета', 'атта', 'аулуа', 'афа', 'аха',
                   'ахаха', 'ахоха', 'ахха', 'аца', 'ацыца', 'ача', 'аша', 'баб', 'бараб', 'батаб', 'бахаб', 'биб',
                   'боб', 'вёв', 'вив', 'вызыв', 'гачаг', 'гег', 'гиг', 'гириг', 'гог', 'гыг', 'гэг', 'дед', 'дид',
                   'диороид', 'довод', 'дойод', 'дород', 'доход', 'дуд', 'еае', 'ейе', 'ёре', 'ере', 'ёте', 'жож',
                   'заказ', 'замаз', 'заз', 'зуз', 'чи', 'и', 'иааи', 'иаи', 'иби', 'иви', 'идииди', 'ижжи', 'изи',
                   'ики', 'или', 'илли', 'иньни', 'ири', 'ирори', 'ихи', 'ичи', 'ичиичи', 'ишши', 'йай', 'йой', 'каак',
                   'кабак', 'кавак', 'казак', 'кайак', 'как', 'канак', 'капак', 'карак', 'касак', 'кассак', 'кек',
                   'кёк', 'келек', 'келлек', 'керек', 'кесек', 'кечёк', 'кибик', 'кижик', 'кизик', 'киик', 'кийик',
                   'кик', 'килик', 'киллилиллик', 'киник', 'киноник', 'кичик', 'кишик', 'ковок', 'кок', 'коллок',
                   'колок', 'комок', 'конок', 'копок', 'коппок', 'корок', 'косок', 'кошок', 'кудук', 'кук', 'кумук',
                   'курук', 'куук', 'кыйык', 'кык', 'кытык', 'кэк', 'кюк', 'лаал', 'лал', 'лобол', 'лол', 'лыл',
                   'мадам', 'мазам', 'макам', 'мам', 'манам', 'марам', 'мелем', 'мем', 'мивим', 'мидим', 'миллим',
                   'мим', 'миним', 'мом', 'моном', 'мум', 'мурум', 'мэм', 'наан', 'набан', 'наган', 'назан', 'накан',
                   'нан', 'напан', 'насан', 'нашан', 'некен', 'нен', 'ненен', 'нигин', 'нимин', 'нойон', 'нон', 'ноон',
                   'норурон', 'нэн', 'нян', 'о', 'обибо', 'обо', 'ово', 'оддо', 'ойо', 'око', 'оло', 'ололо', 'оно',
                   'оо', 'оро', 'ороборо', 'оруро', 'оссо', 'офо', 'очо', 'ошо', 'переп', 'покоп', 'поп', 'посоп',
                   'потоп', 'пуп', 'радар', 'расар', 'ревер', 'реер', 'рейер', 'ремер', 'репер', 'реппер', 'рер',
                   'рогор', 'ророр', 'ротатор', 'ротор', 'рэпєр', 'рэппєр', 'сас', 'секес', 'сиис', 'солос', 'соссос',
                   'статс', 'суккус', 'сукус', 'сус', 'таат', 'такат', 'таннат', 'тартрат', 'тассат', 'тат', 'тауат',
                   'тидит', 'тиллит', 'тимит', 'тирит', 'тит', 'тихит', 'тозот', 'топот', 'торот', 'тумут', 'тут',
                   'тыыт', 'у', 'убу', 'уду', 'улу', 'уруушу', 'фараф', 'феф', 'ханнах', 'хенех', 'хох', 'целец',
                   'чаач', 'чабач', 'чавач', 'чагач', 'чепеч', 'чеч', 'чижич', 'шабаш', 'шалаш', 'шамаш', 'шараш',
                   'шереш', 'шириш', 'шиш', 'шош', 'шугуш', 'шумуш', 'щэщ', 'эвэ', 'эдэ', 'кек', 'лол']
_global_words_1 = ['сарпиночник', 'контрабандист', 'мопед', 'вульгарность', 'ятрышник', 'следопыт', 'оперирование',
                   'шпажист', 'англосаксонец', 'натуралистичность', 'серница', 'раздел', 'памятник', 'антрополог',
                   'новорождённая', 'окрол', 'гальваноскоп', 'кофта', 'председатель', 'ржанище', 'помилованная',
                   'примирение', 'суберин', 'папуаска', 'злободневность', 'эпископ', 'неучтивость', 'адат', 'подавание',
                   'походка', 'хорь', 'брейд-вымпел', 'предпочтение', 'слепушонок', 'кудель', 'эдикт', 'разнеженность',
                   'духанщик', 'вертолётчица', 'светотехника', 'провозгласитель', 'бериллий', 'пискунья', 'отгонщик',
                   'глиптодонт', 'локомобиль', 'пресмыкание', 'старобытность', 'двупланность', 'лютеций', 'прирез',
                   'рявкание', 'перегрузка', 'токсиколог', 'искусительница', 'дикция', 'древность', 'сертификация',
                   'магистраль', 'фагоцитоз', 'всесторонность', 'армада', 'люэс', 'бутоньерка', 'полустишие',
                   'сельхозинвентарь', 'огранка', 'минускул', 'монотипист', 'дань', 'бармен', 'выпирание',
                   'противосияние', 'альтист', 'бекас', 'глиптотека', 'полиграфия', 'уменьшение', 'лункование',
                   'клирос', 'пагода', 'элементарность', 'предпочтительность', 'горицвет', 'ксилофон', 'игиль',
                   'паратость', 'ножовщик', 'гель', 'непроизносимость', 'отшвыривание', 'новолуние', 'обрезок',
                   'технеций', 'самбист', 'инсулин', 'бирманка', 'гвардия', 'папуас', 'оживание', 'заскабливание',
                   'переливт', 'кройка', 'контроверза', 'ниспровержение', 'нагреватель', 'плата', 'паралитик', 'платан',
                   'эндокард', 'скликание', 'инвенция', 'раскутывание', 'загустение', 'чека', 'перенагревание',
                   'припыл', 'тенётчик', 'натюрморист', 'цивилизованность', 'упрощение', 'отопленец', 'свечка',
                   'предплужник', 'юродивая', 'неприличность', 'вех', 'лежание', 'драчливость', 'фидеист', 'дезодорант',
                   'прокапчивание', 'сбережение', 'посыльная', 'фольклористика', 'вдохновитель', 'культурница',
                   'виноградарь', 'пряничник', 'практикант', 'тузлук', 'плач', 'вареник', 'рислинг', 'транш',
                   'укупорщик', 'усложнение', 'фальшивомонетничество', 'пышность', 'подстановка', 'санитар',
                   'линовальщик', 'септик', 'пережидание', 'фалл', 'наивность', 'метафизичность', 'вычищение', 'лярд',
                   'передрессировывание', 'долгожитель', 'метрополитен', 'прошивка', 'подчитывание', 'ёлка',
                   'подкрутка', 'аил', 'концепция', 'обмол', 'обиженная', 'жертвенник', 'отчизна', 'шёпот',
                   'обмыливание', 'водохранилище', 'пантовар', 'притачка', 'кардиография', 'навинчивание',
                   'угнетённость', 'высокопарность', 'ломаная', 'непоследовательность', 'дилетант', 'разгром', 'горло',
                   'коалиция', 'федералист', 'отдыхающий', 'неудовлетворительность', 'театральность', 'шурфование',
                   'подгрузка', 'привкус', 'крольчатина', 'ярка', 'декабристка', 'неоклассик', 'откус', 'педфак',
                   'одежда', 'евпатория', 'индонезия', 'кастрюля', 'качели', 'мамонт', 'копье', 'колледж',
                   'авиаметеостанция', 'гороскоп', 'марево', 'десница', 'мозоль', 'копоть', 'креветка', 'качалка',
                   'конвейер', 'алоэ', 'камбуз', 'катализатор', 'ладонь', 'крыло', 'кий', 'амфибия', 'бородавка',
                   'кафтан', 'стул', 'иордания', 'электричка', 'пещера', 'мундир', 'водоросль', 'бар', 'балерина',
                   'граната', 'брус', 'купальня', 'башмачок', 'берлин', 'жеребец', 'воробей', 'сова', 'леденец',
                   'арена', 'узел', 'софа', 'утюг', 'ландыш', 'вакцина', 'бурьян', 'погреб', 'душ', 'гамбург',
                   'джунгли', 'голень', 'желток', 'лохмотья', 'берег', 'голгофа', 'шкатулка', 'венок', 'малыш',
                   'кемпинг', 'паркет', 'баня', 'департамент', 'боекомплект', 'канзас', 'дренаж', 'капсула',
                   'автомагистраль', 'антиквар', 'мотор', 'карамелька', 'лев', 'впадина', 'декада', 'масленка',
                   'медпункт', 'мультфильм', 'лотерея', 'калория', 'говядина', 'камфара', 'зубок', 'лимузин', 'бильярд',
                   'колдобина', 'иероглиф', 'воск', 'шпаргалка', 'траншея', 'авиастроитель', 'пряник', 'бром',
                   'автопоезд', 'кортик', 'дыхание', 'империя', 'плов']


# сюда копируйте класс WordsDataset, созданный на предыдущем занятии
class WordsDataset(data.Dataset):
    def __init__(self, batch_size=8):  # инициализатор класса
        # здесь код, относящийся к инициализатору
        self.batch_size = batch_size
        _text = "".join(_global_words_0 + _global_words_1).lower()
        self.alphabet = set(_text)  # набор символов
        self.int_to_alpha = dict(enumerate(sorted(self.alphabet)))  # число в символ
        self.alpha_to_int = {b: a for a, b in self.int_to_alpha.items()}  # символ в число

        self.one_hots = torch.eye(len(self.alphabet))

        dataset_0 = [(word.lower(), 0) for word in _global_words_0]
        dataset_1 = [(word.lower(), 1) for word in _global_words_1]
        self.dataset = sorted(dataset_0 + dataset_1, key=lambda x: len(x[0]))

        self.dataset_len = len(self.dataset)

    def __getitem__(self, item):  # формирование и возвращение батча данных по индексу item
        # здесь код, относящийся к __getitem__
        start = self.batch_size * item
        end = min(self.batch_size * (item + 1), self.dataset_len)
        data_set = self.dataset[start: end]
        word_max_length = len(data_set[-1][0])

        x = torch.zeros(end - start, word_max_length, len(self.alphabet))
        y = torch.zeros(end - start, 1)
        empty_letter = torch.zeros(len(self.alphabet))

        for i, (word, answer) in enumerate(data_set):
            for j in range(word_max_length):
                x[i, j] = self.one_hots[self.alpha_to_int[word[j]]] if j < len(word) else empty_letter

            y[i] = answer

        return x, y

    def __len__(self):  # возврат размер обучающей выборки в батчах
        # здесь код, относящийся к __len__
        last = 0 if self.dataset_len % self.batch_size == 0 else 1
        return self.dataset_len // self.batch_size + last


# здесь продолжайте программу
d_train = WordsDataset()
train_data = data.DataLoader(d_train, batch_size=1, shuffle=True)


# здесь объявляйте класс модели
class Model(nn.Module):
    def __init__(self, onehot_size):
        super().__init__()

        self.rnn = nn.RNN(onehot_size, 16, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(32, 1, bias=True)

    def forward(self, x):
        y, h = self.rnn(x)
        h = torch.cat([h[0], h[1]], dim=1)
        return self.linear(h)


# создание модели с числом входов, равным размеру словаря (размеру one-hot векторов)
model = Model(len(d_train.alphabet))

# оптимизатор Adam с шагом обучения 0.01 и параметром weight_decay=0.001
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.001)
loss_func = nn.BCEWithLogitsLoss()  # бинарная кросс-энтропия BCEWithLogitsLoss

epochs = 2  # число эпох обучения (в реальности нужно от 100 и более)
# переведите модель в режим обучения
model.train()

for _e in range(epochs):
    for x_train, y_train in train_data:
        predict = model(x_train.squeeze(dim=0))  # вычислите прогноз модели для x_train
        loss = loss_func(predict, y_train.squeeze(dim=0))  # вычислите потери для predict и y_train

        # выполните один шаг обучения (градиентного спуска)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# переведите модель в режим эксплуатации
model.eval()

Q = 0  # начальное значение доли верных классификаций
for x_train, y_train in train_data:
    with torch.no_grad():
        x_train = x_train.squeeze(0)
        y_train = y_train.squeeze(0)

        p = model(x_train)  # вычислите прогноз модели для x_train
        p = (torch.sign(p) + 1) / 2
        Q += (p.squeeze(0) == y_train).float().mean()  # вычислите долю верных классификаций для p и y_train

Q = (Q / len(d_train)).item()  # усреднение по всем батчам

```
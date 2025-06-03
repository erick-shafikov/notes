/* ИТЕРАТОР
Шаблон — это способ доступа к элементам объекта без раскрытия базового представления.
В этом шаблоне итератор используется для перемещения по контейнеру и обеспечения доступа к элементам контейнера. 
Шаблон подразумевает отделение алгоритмов от контейнера. 
В каких-то случаях алгоритмы, специфичные для этого контейнера, не могут быть отделены.
*/

class RadioStation {
    protected frequency: number;

    constructor(frequency: number) {
        this.frequency = frequency;
    }

    public getFrequency(): number {
        return this.frequency;
    }
}

class StationList {
    protected stations: RadioStation[];
    protected counter: number;

    public addStation(station: RadioStation) {
        this.stations.push(station);
    }

    public removeStation(toRemove: RadioStation) {
        const toRemoveFrequency = toRemove.getFrequency();
        this.stations = this.stations.filter(
            (station) => station.getFrequency() !== toRemoveFrequency
        );
    }

    public count(): number {
        return this.stations.length;
    }

    public current(): RadioStation {
        return this.stations[this.counter];
    }

    public key() {
        return this.counter;
    }

    public next() {
        this.counter++;
    }

    public rewind() {
        this.counter = 0;
    }

    public valid(): boolean {
        return !!this.stations[this.counter];
    }
}

const stationList = new StationList();

stationList.addStation(new RadioStation(89));
stationList.addStation(new RadioStation(101));
stationList.addStation(new RadioStation(102));
stationList.addStation(new RadioStation(103.2));

stationList.removeStation(new RadioStation(89)); // Will remove station 89
export {};

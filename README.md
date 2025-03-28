algo.py: algoritmus (ez a fájl szerintem kész van)

agent.py: egy csúcs, és minden csúcs minden körben meghívja a main-t magára

history_tree.py: minden agent-nek van egy history tree-je

main.py: benne van, amivel ki kéne rajzolni a node-okat

a kapcsolatok valtoznak mar és meg is vannak jelenítve, a history tree-kkel van még munka

## History tree
+ merge trees
+ chop
+ szamlalasi szint meghatározása

### Merge

Ilyen fa van kezdetben:
```
r   r
|   |
0   1
```

Első/második view
```
r   r
|   |
0   1
```
```
    r
  /   \
 0     1
```
Ennek az aljára kell fűzni a csúcs inputját:

Első szemszöge:
```
    r
  /   \
 0     1
 |
 0
```
Második szemszöge:
```
    r
  /   \
 0     1
        |
        1
```

Nagyobb fa merge
```
    r
  /   \
 0     1
|     / |
1     1  0
```
```+```
```
    r
  /   \
 1     0
|     / |
1     1  0
```
```=```
```
    r
  /   \
 1     0
|  \   / |
1   0  1  0
```
És ugye az aljára kéne írni a csúcs inputját. A többi csúcsot meg pirssal összekötni.

A view (V) egy ügynök (p) nézete egy adott körben (t), amely a history tree azon részgráfja, amely az összes legrövidebb utat tartalmazza a gyökértől (r) az adott körben az ügynököt reprezentáló csomópontig (Lt-ben lévő csúcs).

A view-eket küldözgetik egymásnak az ügynökök. Két view merge a cikkben:

![](<Universal Finite-State and Self-Stabilizing Computation in Anonymous Dynamic Networks - Google Chrome 2025. 03. 28. 13_57_09.png>)

Itt szerintem azért van több szín, mert úgy gondolják, hogy nem csak 0 és 1 bemenet van, hanem lehetne 2, 3, ... is.

### Chop

Ezt még nem néztem át, mert ugye a merge kell ahhoz, hogy oldalra nőjön a fa, és az, hogy minden körbe az aljára fűzünk egyet kell ahhoz, hogy lefelé nőjön.

De amúgy a chopban a számlálási szinteket vágjuk le, és csúsztaljuk őket:
```
L0 -> kivágjuk
L1 -> L0
L2 -> L1
L3 -> L2
...
```
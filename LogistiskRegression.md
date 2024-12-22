
### Logistisk regression

Logistisk regression er en metode, der bruges til at klassificere data i kategorier. I Iris-datasættet ønsker vi at forudsige, hvilken af tre blomstertyper (Iris-setosa, Iris-versicolor eller Iris-virginica) en blomst tilhører, baseret på målinger som længde og bredde af kronblade og bægerblade.

Sådan virker det:

#### 1. Linearkombination af data:

Modellen starter med at lave en lineær kombination af de indgående data. Det betyder, at vi vægter hver egenskab med nogle tal (kaldet vægte) og lægger dem sammen med en forskydning (kaldet bias). Formlen ser sådan ud:


$$z_k = w_{k1} \cdot x_1 + w_{k2} \cdot x_2 + \ldots + w_{kn} \cdot x_n + b$$


Her er $$x_1, x_2, \ldots, x_n$$ vores inputegenskaber (fx længde og bredde af kronblad osv), $$w_{k1}, w_{k2}, \ldots, w_{kn}$$ vægte, som modellen lærer, $$k$$ den $$k$$'te klasse
$$b$$ en forskydning, der hjælper med at justere modellen.

For hver klasse i Iris-datasættet beregner modellen et z-værdi (også kaldet logits).

#### 2. Logit og sandsynlighed:

For at forudsige, hvilken klasse en blomst tilhører, skal vi omdanne de lineære kombinationer (z) til sandsynligheder. Til dette bruges logit-funktionen. I binær klassifikation er dette typisk en sigmoidfunktion:


$$\sigma(z) = \frac{1}{1 + e^{-z}}$$


Sigmoidfunktionen skalerer z til en værdi mellem 0 og 1, der kan tolkes som sandsynlighed.

Men fordi vi har flere klasser i Iris-data, bruger vi i stedet softmax-funktionen.

#### 3. Softmax:

Softmax-funktionen bruges til at finde sandsynligheder for flere klasser. Den tager $z{k}-værdierne (en for hver klasse) og normaliserer dem til sandsynligheder, der summerer til 1. Formlen er:


$$P(y = k | x) = \frac{e^{z_k}}{\sum_{j=1}^K e^{z_j}}$$


Hvor $$z_k$$ er logit-værdien for klasse $$k$$, $$K$$ det totale antal klasser $$P(y = k | x)$$ Sandsynligheden for, at data $$x$$ tilhører klasse $$k$$.

### Eksempel med Iris-data:

Forestil dig, at vores model beregner følgende logits (z) for en blomst:


$$z_{\text{setosa}} = 2.5, \quad z_{\text{versicolor}} = 1.2, \quad z_{\text{virginica}} = 0.5$$


#### 1.	Vi beregner eksponentielle værdier:

$$e^{2.5} \approx 12.18, \quad e^{1.2} \approx 3.32, \quad e^{0.5} \approx 1.65$$

#### 2. Beregn softmax-sandsynligheder:

$$\begin{align}
&P(\text{setosa}) &= \frac{12.18}{12.18 + 3.32 + 1.65} \approx 0.72\\
&P(\text{versicolor}) &= \frac{3.32}{12.18 + 3.32 + 1.65} \approx 0.20\\
&P(\text{virginica}) &= \frac{1.65}{12.18 + 3.32 + 1.65} \approx 0.08\\
\end{align}$$

#### 3.	Output:
Modellen forudsiger, at blomsten sandsynligvis er en Iris-setosa, fordi dens sandsynlighed er størst.

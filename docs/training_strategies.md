# Training Strategies — Guía Completa / Complete Guide

> Versión bilingüe: cada concepto en inglés y español.
> Bilingual version: each concept in English and Spanish.

---

## 1. Grokking (Weight Decay)

### English
**What it does:** The model first memorizes all training answers (100% train accuracy, 0% fresh accuracy). Then, weight decay slowly erodes the memorized solution. Over time, only the *algorithm* survives because it's simpler to maintain than a lookup table. The model suddenly generalizes — this is the "grok" moment.

**ELI5:** You memorize multiplication tables but someone erases a little of your notes every night. Eventually, only the understanding of "what multiplication actually means" survives the erasing.

**How it works:**
```
For each training step:
    1. Compute gradient normally
    2. Also shrink all weights by a small factor: w = w × (1 - decay)
    3. Update: w = w - lr × gradient
```
The shrinking (step 2) penalizes large weights. Memorization requires many large weights (one per memorized example). The algorithm requires fewer, smaller weights. Decay makes memorization expensive.

**Our results:** Dominates the top 5 experiments. Weight decay of 0.1 works best. The model memorizes parity quickly, then after thousands of cycles, grokking kicks in and it generalizes.

**Strengths:** Proven, simple, well-understood. Works on algorithmic tasks.
**Weaknesses:** Slow — can take 10-100x longer than memorization to generalize. The "waiting period" wastes compute.

---

### Español
**Qué hace:** El modelo primero memoriza todas las respuestas de entrenamiento (100% en entrenamiento, 0% en datos nuevos). Luego, el "weight decay" (decaimiento de pesos) erosiona lentamente la solución memorizada. Con el tiempo, solo sobrevive el *algoritmo* porque es más simple de mantener que una tabla de búsqueda. El modelo de repente generaliza — este es el momento "grok".

**Explicado para un niño de 5 años:** Te memorizas las tablas de multiplicar pero alguien borra un poquito de tus apuntes cada noche. Al final, lo único que sobrevive es tu *comprensión* de "qué significa realmente multiplicar".

**Cómo funciona:**
```
En cada paso de entrenamiento:
    1. Calcular el gradiente normalmente
    2. También encoger todos los pesos un poco: w = w × (1 - decaimiento)
    3. Actualizar: w = w - lr × gradiente
```
El encogimiento (paso 2) penaliza pesos grandes. La memorización requiere muchos pesos grandes (uno por cada ejemplo memorizado). El algoritmo requiere menos pesos, más pequeños. El decaimiento hace que memorizar sea "caro".

**Nuestros resultados:** Domina los 5 mejores experimentos. Un decaimiento de 0.1 funciona mejor. El modelo memoriza paridad rápidamente, y después de miles de ciclos, el grokking se activa y generaliza.

**Fortalezas:** Probado, simple, bien entendido. Funciona en tareas algorítmicas.
**Debilidades:** Lento — puede tomar 10-100x más tiempo que la memorización para generalizar. El "período de espera" desperdicia cómputo.

---

## 2. PerpGrad (Gradiente Perpendicular / ⊥Grad)

### English
**What it does:** After memorization, the optimizer "cheats" — instead of learning new things, it just makes the model more confident about what it already knows (scaling logits up). This is called Naive Loss Minimization (NLM). PerpGrad removes this cheating component from the gradient. Every gradient step must actually change what the model computes, not just amplify it.

**ELI5:** You're learning to ride a bike. But instead of practicing balance, you keep making the bike bigger (you feel more stable but you're not). PerpGrad says: "stop making the bike bigger — only practice balance."

**How it works:**
```
For each training step:
    1. Compute gradient normally
    2. Project out the component along the weight direction:
       grad_perp = grad - (w·grad)/(w·w) × w
    3. Update: w = w - lr × grad_perp
```
The projection (step 2) removes the part of the gradient that just scales weights up. What's left is the part that actually changes the model's behavior.

**Our results:** exp_095 (d=48, PerpGrad) reached #3 on the leaderboard at 11.4%. Competitive with grokking but with fewer experiments exploring it.

**Strengths:** No waiting period — prevents NLM from the start. Works without weight decay.
**Weaknesses:** Can be too aggressive — removes gradient signal that might be useful. Less proven than weight decay.

---

### Español
**Qué hace:** Después de memorizar, el optimizador "hace trampa" — en vez de aprender cosas nuevas, solo hace que el modelo esté más seguro de lo que ya sabe (escalando los logits). Esto se llama Minimización Ingenua de Pérdida (NLM). PerpGrad elimina esta trampa del gradiente. Cada paso del gradiente debe realmente cambiar lo que el modelo computa, no solo amplificarlo.

**Explicado para un niño de 5 años:** Estás aprendiendo a andar en bicicleta. Pero en vez de practicar el equilibrio, sigues haciendo la bicicleta más grande (te sientes más estable pero no lo eres). PerpGrad dice: "deja de hacer la bici más grande — solo practica el equilibrio."

**Cómo funciona:**
```
En cada paso de entrenamiento:
    1. Calcular el gradiente normalmente
    2. Proyectar y eliminar el componente en la dirección del peso:
       grad_perp = grad - (w·grad)/(w·w) × w
    3. Actualizar: w = w - lr × grad_perp
```
La proyección (paso 2) elimina la parte del gradiente que solo escala los pesos. Lo que queda es la parte que realmente cambia el comportamiento del modelo.

**Nuestros resultados:** exp_095 (d=48, PerpGrad) alcanzó el #3 en la tabla de posiciones con 11.4%. Competitivo con grokking pero con menos experimentos explorándolo.

**Fortalezas:** Sin período de espera — previene NLM desde el inicio. Funciona sin decaimiento de pesos.
**Debilidades:** Puede ser demasiado agresivo — elimina señal de gradiente que podría ser útil. Menos probado que weight decay.

---

## 3. StableMax (nuestra función de pérdida actual / our current loss function)

### English
**What it does:** Replaces the exponential function in softmax with a linear ramp. Standard softmax can "collapse" — when logits get very large, floating-point math breaks down, gradients go to zero, and learning stops forever. StableMax prevents this by growing linearly instead of exponentially.

**ELI5:** Normal softmax is like a balloon — it inflates faster and faster until it pops (gradients die). StableMax is like a ruler — it grows steadily and never pops.

**Our results:** All experiments currently use StableMax. It prevents gradient collapse but may give weaker gradient signals than regular softmax for easy decisions (like choosing S vs D).

---

### Español
**Qué hace:** Reemplaza la función exponencial en softmax con una rampa lineal. El softmax estándar puede "colapsar" — cuando los logits se hacen muy grandes, la aritmética de punto flotante se rompe, los gradientes van a cero, y el aprendizaje se detiene para siempre. StableMax previene esto creciendo linealmente en vez de exponencialmente.

**Explicado para un niño de 5 años:** El softmax normal es como un globo — se infla cada vez más rápido hasta que explota (los gradientes mueren). StableMax es como una regla — crece de forma estable y nunca explota.

**Nuestros resultados:** Todos los experimentos usan StableMax actualmente. Previene el colapso de gradientes pero puede dar señales de gradiente más débiles que softmax regular para decisiones fáciles (como elegir S vs D).

---

## 4. Focal Loss (disponible pero no usada aún / available but not yet used)

### English
**What it does:** Regular cross-entropy treats all mistakes equally. Focal loss says: "if you're already getting something right, spend less energy on it. Focus on the hard examples." It upweights mistakes and downweights easy wins.

**ELI5:** Your teacher notices you already know addition but struggle with subtraction. Instead of practicing both equally, the teacher gives you mostly subtraction problems. That's focal loss.

**Formula:** `loss = (1 - p_correct)^γ × normal_loss`

When `p_correct` is high (easy example), the `(1-p)^γ` factor shrinks the loss. When `p_correct` is low (hard example), the loss stays large.

**Why we need it:** The DAME bug — the model always predicts 'D' because it's easy. DIFF examples have low loss (already correct). SAME examples have high loss (always wrong). Focal loss would amplify the SAME loss and force the model to learn the distinction.

**Our status:** Available in the code, mutation can select it, but no experiment has been born with it yet.

---

### Español
**Qué hace:** La entropía cruzada regular trata todos los errores igual. Focal loss dice: "si ya estás acertando algo, gasta menos energía en eso. Enfócate en los ejemplos difíciles." Amplifica los errores y reduce las victorias fáciles.

**Explicado para un niño de 5 años:** Tu maestro nota que ya sabes sumar pero te cuesta restar. En vez de practicar ambos por igual, el maestro te da mayormente problemas de resta. Eso es focal loss.

**Fórmula:** `pérdida = (1 - p_correcto)^γ × pérdida_normal`

Cuando `p_correcto` es alto (ejemplo fácil), el factor `(1-p)^γ` encoge la pérdida. Cuando `p_correcto` es bajo (ejemplo difícil), la pérdida se mantiene grande.

**Por qué lo necesitamos:** El bug DAME — el modelo siempre predice 'D' porque es fácil. Los ejemplos DIFF tienen pérdida baja (ya correctos). Los ejemplos SAME tienen pérdida alta (siempre incorrectos). Focal loss amplificaría la pérdida de SAME y forzaría al modelo a aprender la distinción.

**Nuestro estado:** Disponible en el código, la mutación puede seleccionarlo, pero ningún experimento ha nacido con él aún.

---

## 5. Lo que nos falta / What we're missing

### SAM (Sharpness-Aware Minimization)

**English:** Instead of finding just any low point in the loss landscape, SAM finds *wide flat* valleys. A wide valley means small changes to the input don't change the output much — that's generalization. A narrow valley means the model is fragile.

**Español:** En vez de encontrar cualquier punto bajo en el paisaje de pérdida, SAM encuentra *valles anchos y planos*. Un valle ancho significa que pequeños cambios en la entrada no cambian mucho la salida — eso es generalización. Un valle angosto significa que el modelo es frágil.

**ELI5 / Para un niño:** Imagina que estás parado en una montaña y necesitas encontrar un lago. SAM no solo busca cualquier charco — busca un lago grande y plano. Si el viento sopla un poco, sigues en el lago. En un charco, el viento te saca.

**Status:** Not implemented. Would add ~10 lines of code. Could be a strong contender.

---

### Warm Restarts (Reinicios con Temperatura)

**English:** Periodically reset the learning rate back to high. The model escapes wherever it's stuck and settles somewhere new, hopefully better. Like shaking a snow globe — everything redistributes.

**Español:** Periódicamente reiniciar la tasa de aprendizaje a un valor alto. El modelo escapa de donde esté atascado y se asienta en un lugar nuevo, esperemos que mejor. Como agitar una bola de nieve — todo se redistribuye.

**ELI5 / Para un niño:** Estás armando un rompecabezas y te atascaste. En vez de seguir forzando piezas, mezclas todo y empiezas de nuevo con piezas más grandes. A veces encuentras una mejor solución.

**Status:** Not implemented. Simple to add.

---

### Label Smoothing (Suavizado de Etiquetas)

**English:** Instead of training the model to say "the answer is 100% S", train it to say "the answer is 90% S and 10% everything else." Prevents overconfidence. The model learns to be calibrated rather than extreme.

**Español:** En vez de entrenar al modelo para que diga "la respuesta es 100% S", entrenarlo para que diga "la respuesta es 90% S y 10% todo lo demás." Previene el exceso de confianza. El modelo aprende a ser calibrado en vez de extremo.

**ELI5 / Para un niño:** Tu maestro nunca dice "estás COMPLETAMENTE seguro de que es S." Siempre dice "probablemente es S, pero mantente abierto." Así aprendes a pensar en vez de solo adivinar.

**Status:** Not implemented. Would directly help with the DAME bug — the model was 100% confident about D. With label smoothing, it would maintain uncertainty and be more willing to switch to S.

---

### Lion Optimizer

**English:** Google's optimizer that uses the sign of momentum (±1) instead of its magnitude. Uses 50% less memory than Adam and often converges faster on small models. Each weight gets pushed in the direction it's been going, with uniform strength.

**Español:** El optimizador de Google que usa el signo del momentum (±1) en vez de su magnitud. Usa 50% menos memoria que Adam y a menudo converge más rápido en modelos pequeños. Cada peso se empuja en la dirección que ha estado yendo, con fuerza uniforme.

**ELI5 / Para un niño:** Adam empuja cada pieza del rompecabezas con diferente fuerza (algunas fuerte, otras suave). Lion empuja todas las piezas con la misma fuerza, pero en la dirección correcta. A veces eso funciona mejor.

**Status:** Not implemented. Would need ~20 lines. Could be a good mutation option.

---

### Noise Injection (Inyección de Ruido)

**English:** Add random noise to weights every N steps. Forces the model out of local minima. Like weight decay but more random — instead of systematically shrinking, you shake everything randomly.

**Español:** Agregar ruido aleatorio a los pesos cada N pasos. Fuerza al modelo a salir de mínimos locales. Como weight decay pero más aleatorio — en vez de encoger sistemáticamente, sacudes todo al azar.

**ELI5 / Para un niño:** Estás caminando por un laberinto y te atascaste. Alguien te sacude un poco al azar. A veces aterrizas en un camino mejor.

**Status:** Not implemented. Very simple to add.

---

## 6. Resumen comparativo / Comparative Summary

| Strategy | Speed | Stability | Best for | Status |
|----------|-------|-----------|----------|--------|
| Grokking (wd=0.1) | Slow start, strong finish | Stable | Algorithmic tasks | ★ **Leader** |
| PerpGrad | Fast start | Less stable | Quick exploration | **#3 contender** |
| StableMax | — (loss fn) | Very stable | Preventing collapse | **Used by all** |
| Focal Loss | Fast on hard examples | Stable | Imbalanced outputs | Available, untested |
| SAM | Medium | Very stable | Generalization | **Not implemented** |
| Warm Restarts | Fast recovery | Unstable | Escaping plateaus | **Not implemented** |
| Label Smoothing | Medium | Stable | Overconfidence | **Not implemented** |
| Lion | Fast | Stable | Small models | **Not implemented** |
| Noise Injection | Medium | Unstable | Escaping minima | **Not implemented** |

---

## 7. Recomendación / Recommendation

Las tres estrategias que más impacto tendrían si las implementamos ahora:

The three strategies that would have the most impact if implemented now:

1. **SAM** — directly targets generalization, complements grokking
2. **Label Smoothing** — directly fixes the overconfidence/DAME bug
3. **Lion** — faster convergence on small models, uses less memory

Todas pueden ser mutaciones en nuestro algoritmo genético. La evolución decidirá cuál gana.

All can be mutations in our genetic algorithm. Evolution will decide which wins.

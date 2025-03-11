### **Comparación entre Adagrad y Adadelta**  

Ambos algoritmos son optimizadores adaptativos, pero tienen diferencias clave:  

| **Característica** | **Adagrad** | **Adadelta** |
|------------------|------------|------------|
| **Tasa de aprendizaje** | Disminuye continuamente debido a la acumulación de gradientes. Puede volverse demasiado pequeña y detener el aprendizaje. | No requiere una tasa de aprendizaje fija, ya que ajusta automáticamente la magnitud de las actualizaciones. |
| **Acumulación de gradientes** | Suma de los cuadrados de los gradientes, lo que puede provocar una reducción extrema en la tasa de aprendizaje. | Usa un promedio exponencial de los gradientes pasados para mantener una actualización estable. |
| **Ventajas** | Bueno para problemas convexos y parámetros con escalas muy diferentes. | Evita la disminución extrema de la tasa de aprendizaje, lo que mejora la convergencia en problemas más complejos. |

### **Implementación en este programa**  
1. **C++ (Pybind11)**
   - Se creó una clase `Adadelta` con tres vectores: pesos, acumulador de gradientes y acumulador de actualizaciones.
   - La actualización de pesos usa un promedio exponencial de los gradientes en lugar de acumularlos indefinidamente como en Adagrad.
   - Se expuso la funcionalidad a Python mediante Pybind11.

2. **Python (Script)**
   - Se generaron datos de regresión lineal.
   - Se ejecutó Adadelta para optimizar los parámetros sin una tasa de aprendizaje fija.
   - Se graficó la evolución de la pérdida y el ajuste del modelo.

Adadelta mejora sobre Adagrad al evitar que la tasa de aprendizaje disminuya demasiado rápido, lo que lo hace más robusto en optimización a largo plazo.

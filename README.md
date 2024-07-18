## Overview
В этом репозитории реализован метод Kernel Change Point (KCP) для обнаружения точек изменения корреляции в 
многомерных временных рядах.


## Reference
Метод основан на работе, описанной в статье:

Jedelyn Cabrieto, Francis Tuerlinckx, Peter Kuppens, 
Borbála Hunyadi & Eva Ceulemans (2017).
**Testing for the Presence of Correlation Changes in a Multivariate Time Series: A
Permutation Based Approach.**

## Установка
1. Клонирование репозитория:

```git clone https://github.com/kuznetsona/kcp_test.git```

2. Создание виртуального окружения:

```python -m venv venv```

3. Активация виртуального окружения:

```source venv/bin/activate```

4. Установка зависимостей:

```pip install -r requirements.txt```

5. Запуск кода:

Чтобы сгенерировать синтетические временноq ряд с 
известными точками изменения, запустите скрипт `test_kcp_method.py`. 
Этот скрипт создаст синтетический набор данных и выведет известные точки изменения.

```python test_kcp_method.py```


## Описание методов:
### Методы в main_kcp_method.py:

1. `running_correlations()` - вычисляет текущие корреляции для заданного 
временного ряда (преобразованные с помощью преобразования Фишера Z);

2. `compute_kernel_matrix()` - вычисляет матрицу ядра на основе текущих корреляций;

3. `within_phase_variance()` - вычисляет дисперсию внутри фазы для заданного интервала;

4. `elbow_method()` - реализация метода "elbow" для определения оптимального числа точек изменения;

5. `kcp_detection()` - выполняет обнаружение точек изменения с использованием метода KCP;

6. `permutation_test()` - выполняет тест значимости на основе перестановок для обнаружения точек изменения;

7. `plot_time_series()` -> - строит временные ряды и текущие корреляции с реальными и предсказанными точками изменения;

8. `plot_min_variances()` - строит график минимальных дисперсий внутри фаз для каждого количества точек изменения. Для метода `elbow_method()`;

9. `print_summary()` - вывод результатов обнаружения точек изменения.





#include "head.h"
int fCalcCount;
real A = -10, B = 10;
vector1D xExact = { 5.99926, 8.00039 };
random_device rd; // гениратор простых числе
mt19937_64 e2(1); // сиииид
uniform_real_distribution<> dist(A, B); // гениратор вешественных рандомных в диапозане [A,B]
vector<real> Es = { 1, 5e-1, 1e-1, 5e-2, 1e-2 }; // точность
double P = 0.1; //вероятность (смотри метод table1)
int m = 10; //количество итераций в методе (смотри метод table2)

//------------------------------------------------------------------------------
struct methodResult
{
	real E;
	int iterationsCount;
	int fCalcCount;
	vector1D x0, x, xPrev, S, gradf;
	matrix2D A;
	real fx, fxPrev, lambda;
	void printResult(std::ofstream& fout);
};

// Вывод результатов в поток
void methodResult::printResult(std::ofstream& fout) {
	fout << iterationsCount << "\t"
		<< fCalcCount << "\t";
	printTeXVector(fout, x);
	fout << "\t" << fx << endl;
}



//------------------------------------------------------------------------------
// Функция для исследования min=0 в точке (1,1)
// x - вектор аргументов функции
inline real f(const vector1D& x) {
	fCalcCount++;
	real C[6] = { 2, 4, 2, 6, 2, 3 };
	real a[6] = { -3, -6, 2, 6, -3, 8 };
	real b[6] = { 6, -8, -8, 8, -4, -1 };

	real result = 0;
	for (size_t i = 0; i < 6; i++)
		result += C[i] / (1 + pow((x[0] - a[i]), 2) + pow((x[1] - b[i]), 2));
	return -result;
}

// Поиск интервала, содержащего минимум
// f - целевая одномерная функция
// a,b - искомые границы отрезка
// x - начальное приближение
// S - орты (направления)
void interval(const function<real(const vector1D& x)>& f, real& a, real& b, vector1D& x, vector1D& S)
{
	real lambda0 = 0.0;
	real delta = 1.0e-8;
	real lambda_k_minus_1 = lambda0;
	real f_k_minus_1 = f(x + S * lambda_k_minus_1);
	real lambda_k;
	real f_k;
	real lambda_k_plus_1;
	real f_k_plus_1;
	real h;
	if (f(x + S * lambda0) > f(x + S * (lambda0 + delta)))
	{
		lambda_k = lambda0 + delta;
		h = delta;
	}
	else
	{
		lambda_k = lambda0 - delta;
		h = -delta;
	}
	f_k = f(x + S * lambda_k);
	while (true)
	{
		h *= 2.0;
		lambda_k_plus_1 = lambda_k + h;
		f_k_plus_1 = f(x + S * lambda_k_plus_1);
		if (f_k > f_k_plus_1)
		{
			lambda_k_minus_1 = lambda_k;
			f_k_minus_1 = f_k;
			lambda_k = lambda_k_plus_1;
			f_k = f_k_plus_1;
		}
		else
		{
			a = lambda_k_minus_1;
			b = lambda_k_plus_1;
			if (b < a)
				swap(a, b);
			return;
		}
	}
}

// Вычисление n-го числа Фибоначии
inline real fib(int n)
{
	real sqrt5 = sqrt(5.0), pow2n = pow(2.0, n);
	return (pow(1.0 + sqrt5, n) / pow2n - pow(1.0 - sqrt5, n) / pow2n) / sqrt5;
}

// Определение коэффициента лямбда методом Фибоначчи
// f - минимизируемая функция
// x - начальное значение
// S - базис
// E - точность
real fibonacci(const function<real(const vector1D& x)>& f, vector1D& x, vector1D& S, real E)
{
	real a, b;
	interval(f, a, b, x, S);
	int iter;
	real len = fabs(a - b);
	int n = 0;
	while (fib(n) < (b - a) / E) n++;
	iter = n - 3;
	real lambda1 = a + (fib(n - 2) / fib(n)) * (b - a);
	real f1 = f(x + S * lambda1);
	real lambda2 = a + (fib(n - 1) / fib(n)) * (b - a);
	real f2 = f(x + S * lambda2);
	for (int k = 0; k < n - 3; k++)
	{
		if (f1 <= f2)
		{
			b = lambda2;
			lambda2 = lambda1;
			f2 = f1;
			lambda1 = a + (fib(n - k - 3) / fib(n - k - 1)) * (b - a);
			f1 = f(x + S * lambda1);
		}
		else
		{
			a = lambda1;
			lambda1 = lambda2;
			f1 = f2;
			lambda2 = a + (fib(n - k - 2) / fib(n - k - 1)) * (b - a);
			f2 = f(x + S * lambda2);
		}
		len = b - a;
	}
	lambda2 = lambda1 + E;
	f2 = f(x + S * lambda2);
	if (f1 <= f2)
		b = lambda1;
	else
		a = lambda1;
	return (a + b) / 2.0;
}

// Метод Розенброка
// f - оптимизиуремая функция
// x0 - начальное приближение
// E - точность
// funcname - название функции
methodResult calcByRosenbrock(
	const function<real(const vector1D& x)>& f,
	const vector1D& x0,
	real E,
	const string& funcname)
{
	ofstream fout("table" + funcname + ".txt");
	ofstream steps("_" + funcname + ".txt");
	fout << scientific;
	steps << fixed << setprecision(12);
	steps << x0 << endl;
	fout << "scientific " << "iterationsCount " << "\t"
		  << "fCalcCount " << "\t"
		  << "x " << "\t"
		  << "f(x)" << endl;

	methodResult result;
	//fCalcCount = 0;
	vector1D xPrev, B, x = x0;
	int maxiter = 1000;
	int iterationsCount = 0, count = 0;
	real lambda1, lambda2;
	matrix2D A(2);
	A[0] = { 1.0, 0.0 };
	A[1] = { 0.0, 1.0 };

	// Начальные ортогональные направления
	matrix2D S(2);
	S[0] = { 1.0, 0.0 };
	S[1] = { 0.0, 1.0 };

	do {
		xPrev = x;

		// Минимализируем функцию в направлениях S^k_1...S^k_n
		lambda1 = fibonacci(f, x, S[0], E);
		x = x + S[0] * lambda1;
		lambda2 = fibonacci(f, x, S[1], E);
		x = x + S[1] * lambda2;

		// Построение новых ортогональных направлений при
		// сортировке лямбд в порядке убывания по абсолютным значениям
		A[0] = S[0] * lambda1 + S[1] * lambda2;
		if (fabs(lambda1 >= lambda2))
			A[1] = S[1] * lambda2;
		else
			A[1] = S[0] * lambda1;


		// Ортогонализация Грамма-Шмидта
		S[0] = A[0] / calcNormE(A[0]);
		B = A[1] - S[1] * A[1] * S[1];
		if (calcNormE(B) > E)
			S[1] = B / calcNormE(B);
		iterationsCount++;

		fout << iterationsCount << "\t"
			<< fCalcCount << "\t"
			<< x << "\t"
			<< f(x) << endl;

		steps << x << endl;

	} while (abs(f(x) - f(xPrev)) > E && abs(calcNormE(x- xPrev)) > E && iterationsCount < maxiter);

	fout.close();
	steps.close();

	result.E = E;
	result.iterationsCount = iterationsCount;
	result.fCalcCount = fCalcCount;
	result.x0 = x0;
	result.x = x;
	result.fx = f(x);

	return result;
}

//------------------------------------------------------------------------------
// Исследования
//------------------------------------------------------------------------------

// Метод простого случайного поиска
int simpleRandomSearch(real E, real P, bool doVisualisation) {

	cout << "Simple random search method" << endl;
	real V = pow((B - A), 2);
	real V_E = E * E;
	real P_E = V_E / V;
	real N = log(1.0 - P) / log(1.0 - P_E);
	cout << N << endl;


	vector1D x1, x2, minX = { dist(e2), dist(e2) };
	real fx1, fx2, fxMin = f(minX);

	ofstream fout("simpleRandomSearch.txt");
	for (size_t i = 0; i < N; i++)
	{
		x1 = { dist(e2), dist(e2) };
		x2 = { dist(e2), dist(e2) };
		fx1 = f(x1);
		fx2 = f(x2);
		if (doVisualisation) {
			if (fx1 < fxMin && fx1 < fx2) {
				fxMin = fx1;
				minX = x1;
				fout << minX << endl;
			}
			else if (fx2 < fxMin && fx2 < fx1) {
				fxMin = fx2;
				minX = x2;
				fout << minX << endl;
			}
		}
		else {
			if (fx1 < fxMin && fx1 < fx2) {
				fxMin = fx1;
				minX = x1;
			}
			else if (fx2 < fxMin && fx2 < fx1) {
				fxMin = fx2;
				minX = x2;
			}
		}
	}
	//cout << minX;
	fout <<"("<< minX[0]<<";"<<minX[1]<<")" << "\n";
	// Визуализация
	if (doVisualisation) {
		string runVisualisation = "python plot.py";
		system(runVisualisation.c_str());
	}
	fout.close();
	return N;
}

void table1() {

	ofstream fout("table1.txt");
	fout << "P="<< P << endl;
		for (auto E : Es)
		{
			fout <<"E=" << E << " N=" << simpleRandomSearch(E, P, false) << "\n";
		}
		fout << endl;
	fout.close();
}

// 1-ый алгоритм
pair <int, double> alg1(int m) {

	fCalcCount = 0;
	methodResult res;
	vector1D x, minX = { dist(e2), dist(e2) };
	real fxMin = f(minX);
	real E = 1e-12;
	int count = 0;
	while (count < m) {
		x = { dist(e2), dist(e2) };
		res = calcByRosenbrock(f, x, E, "alg1");
		if (res.fx < fxMin) {
			minX = res.x;
			fxMin = res.fx;
			count = 0;
		}
		else count++;
	}
	return make_pair(fCalcCount, calcNormE(res.x - xExact));
}

// 2-ой алгоритм
pair <int, double> alg2(int m) {

	fCalcCount = 0;
	methodResult res;
	vector1D x, minX = { dist(e2), dist(e2) };
	real fxMin = f(minX), fx;
	real E = 1e-12;
	int count = 0;
	while (count < m) {
		do {
			x = { dist(e2), dist(e2) };
			fx = f(x);
			count++;
		} while (fx > fxMin && count < m);

		if (count < m) {
			res = calcByRosenbrock(f, x, E, "alg2");
			if (res.fx < fxMin) {
				minX = res.x;
				fxMin = res.fx;
				count = 0;
			}
		}
	}
	return make_pair(fCalcCount, calcNormE(res.x - xExact));
}

// 3-ий алгоритм
pair <int, double> alg3(int m) {

	fCalcCount = 0;
	methodResult res;
	vector1D x = { dist(e2), dist(e2) }, minX = x;
	real fxMin = f(minX);
	real E = 1e-12;
	int count = 0;
	while (count < m) {
		res = calcByRosenbrock(f, x, E, "alg3");
		x = { dist(e2), dist(e2) };
		int step = 1;
		vector1D xTmp = res.x + (res.x - x) * step;
		real fxTmp = f(xTmp);
		while (fxTmp >= res.fx &&
			A < xTmp[0] && xTmp[0] < B &&
			A < xTmp[1] && xTmp[1] < B)
		{
			xTmp = res.x + (res.x - x) * step;
			fxTmp = f(xTmp);
			step++;
			count++;
		}

		if (res.fx < fxMin) {
			minX = res.x;
			fxMin = res.fx;
			count = 0;
		}
	}
	return make_pair(fCalcCount, calcNormE(res.x - xExact));
}

void table2() {
	pair <int, real> result;
	ofstream fout("table2.txt");
		fout << "iter:" << m << "\n";
		result = alg1(m);
		fout << "fCalcCount= " << result.first << " " << "|x-x*| = " << result.second << "\n";
		result = alg2(m);
		fout << "fCalcCount= " << result.first << " " << "|x-x*| = " << result.second << "\n";
		result = alg3(m);
		fout << "fCalcCount= " << result.first << " " << "|x-x*| = " << result.second << "\n";
	fout.close();
}

void main()
{
	table1();
	table2();
}
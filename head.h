#pragma once
#define _CRT_SECURE_NO_WARNINGS
#define UNICODE
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <functional>
#include <cmath>
#include <math.h>
#include <random>
#include <algorithm>

using namespace std;


// float || double
typedef double real;
typedef vector <real> vector1D;
typedef vector <vector <real>> matrix2D;



// ��������� �� ���������
inline bool operator==(const vector1D& a, const vector1D& b) {
#ifdef _DEBUG
	if (a.size() != b.size())
		throw std::exception();
#endif
	for (int i = 0; i < a.size(); ++i)
		if (a[i] != b[i])
			return false;

	return true;
}


// �������� ��������
inline vector1D operator+(const vector1D& a, const vector1D& b) {
#ifdef _DEBUG
	if (a.size() != b.size())
		throw std::exception();
#endif
	vector1D result = a;
	for (int i = 0; i < b.size(); i++)
		result[i] += b[i];
	return result;
}


// �������� ������
inline matrix2D operator+(const matrix2D& a, const matrix2D& b) {
#ifdef _DEBUG
	if (a.size() != b.size())
		throw std::exception();
#endif
	matrix2D result = a;
	for (int i = 0; i < b.size(); i++)
		for (int j = 0; j < b.size(); j++)
			result[i][j] += b[i][j];
	return result;
}


// �������� ������
inline matrix2D operator/(const matrix2D& a, const real& b) {

	matrix2D result = a;
	for (int i = 0; i < a.size(); i++)
		for (int j = 0; j < a.size(); j++)
			result[i][j] /= b;
	return result;
}


// ��������� ��������
inline vector1D operator-(const vector1D& a, const vector1D& b) {
#ifdef _DEBUG
	if (a.size() != b.size())
		throw std::exception();
#endif
	vector1D result = a;
	for (int i = 0; i < b.size(); i++)
		result[i] -= b[i];
	return result;
}


inline vector1D operator-(const vector1D& a) {
	vector1D result = a;
	for (int i = 0; i < a.size(); i++)
		result[i] = -result[i];
	return result;
}


// ��������� ������� �� ������
inline vector1D operator*(const matrix2D& a, const vector1D& b) {
	vector1D result = { 0.0, 0.0 };
	for (int i = 0; i < a.size(); i++)
		for (int j = 0; j < a.size(); j++)
			result[i] += a[i][j] * b[j];
	return result;
}


// ��������� �� ���������
inline vector1D operator*(const vector1D& a, double b) {
	vector1D result = a;
	for (int i = 0; i < result.size(); i++)
		result[i] *= b;
	return result;
}


// ��������� �� ���������
inline vector1D operator*(double b, const vector1D& a) {
	return operator*(a, b);
}


// ������� �� ���������
inline vector1D operator/(const vector1D& a, double b) {
	vector1D result = a;
	for (int i = 0; i < result.size(); i++)
		result[i] /= b;
	return result;
}


// ������� �� ���������
inline vector1D operator/(double b, const vector1D& a) {
	return operator/(a, b);
}


// ��������� ������������
inline real operator*(const vector1D& a, const vector1D& b) {
#ifdef _DEBUG
	if (a.size() != b.size())
		throw std::exception();
#endif
	real sum = 0;
	for (int i = 0; i < a.size(); i++)
		sum += a[i] * b[i];
	return sum;
}


// ��������� ����� �������
inline std::ostream& operator<<(std::ostream& out, const vector1D& v) {

	for (int i = 0; i < v.size() - 1; ++i)
		out << v[i] << " ";
	out << v.back();
	return out;
}

// ��������� ����� ������� ��� TeX
inline void printTeXVector(std::ofstream& fout, const vector1D& v) {
	fout << "(";
	for (int i = 0; i < v.size() - 1; ++i)
		fout << v[i] << ", ";
	fout << v.back() << ")^T";
}

// ��������� ����� �������
inline std::ostream& operator<<(std::ostream& out, const matrix2D& v) {
	for (int i = 0; i < v.size() - 1; ++i)
		out << v[i] << "  ";
	out << v.back();
	return out;
}

// ��������� �����
real calcNormE(const vector1D& x) {
	return sqrt(x * x);
}


// ������������ �������
real det(const matrix2D& m) {
	return m[0][0] * m[1][1] - m[0][1] * m[1][0];
}
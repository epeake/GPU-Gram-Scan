#include <iostream>
#include <fstream>
#include <string>
#include <errno.h>
#include <vector>
#include "test.h"

using namespace std;

std::vector<Point> getData(string filename){
  std::vector<Point> data;

  string STRING;
  ifstream infile;
  infile.open(filename);
  if (errno != 0) {
    perror("infile.open");
    exit(EXIT_FAILURE);
  }
  
  while (!infile.eof()) {
    getline(infile, STRING);
    if (infile.fail()) {
      perror("getline");
      exit(EXIT_FAILURE);
    }
    int comma = STRING.find(',');
    string first_num = STRING.substr(0, comma);
    string second_num = STRING.substr(comma + 1, STRING.length());

    Point current_point;
    current_point.x = stoi(first_num);
    current_point.y = stoi(second_num);

    data.push_back(current_point);
    cout << STRING << '\n';
  }

  infile.close();
  if (infile.fail()) {
      perror("infile.close");
      exit(EXIT_FAILURE);
  }
  return data;
}

int main() {
  Point p1;
  p1.x = 1;
  p1.y = 2;

  
  // system ("pause");
  cout << p1.x << ' ' << p1.y << '\n';
  getData("test-data/test1.in");
}

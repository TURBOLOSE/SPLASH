#pragma once

#include <vector>
#include <numeric>

template <typename num_t>
class vector3d :public std::array<num_t, 3>
{
public:
    using std::array<num_t, 3>::operator=;
    using std::array<num_t, 3>::operator[];

 constexpr vector3d& operator+=(const vector3d& right) {

        std::transform(this->begin(), this->end(), right.begin(), this->begin(), std::plus<num_t>());
        return *this;
    };

    constexpr vector3d operator+(const vector3d& right) {

        vector3d result = *this;
        return result += right;
    };


    constexpr vector3d& operator-=(const vector3d& right) {

        std::transform(this->begin(), this->end(), right.begin(), this->begin(), std::minus<num_t>());
        return *this;
    };

    constexpr vector3d operator-(const vector3d& right) {

        vector3d result = *this;
        return result -= right;
    };


    constexpr vector3d& operator*=(const num_t number) { //by scalar

        std::transform(this->begin(), this->end(), this->begin(), [&number](auto& c){return c*number;});
        return *this;
    };


    constexpr vector3d operator*(const num_t number) {

        vector3d result = *this;
        return result *= number;
    };

       constexpr vector3d& operator*=(const vector3d& right) { //by other vector

        std::transform(this->begin(),  this->end(), right.begin(), this->begin(), std::multiplies<num_t>());
        return *this;
    };

    constexpr vector3d operator*(const vector3d& right) {

        vector3d result = *this;
        return result *= right;
    };

    constexpr vector3d& operator/=(const num_t number) {  //by scalar

        std::transform( this->begin(), this->end(), this->begin(),  [&number](num_t& c){return c/number;});
        return *this;
    };

    constexpr vector3d operator/(const num_t number) {

        vector3d result = *this;
        return result /= number;
    };


    constexpr num_t norm(){
    
        num_t result;
        vector3d temp;
        temp = *this * *this;
        //result = std::reduce(temp.begin(), temp.end());
        result=temp[0]+temp[1]+temp[2];
        return std::sqrt(result);
    
    };


    void print(){
        std::cout<<"("<<(*this)[0]<<","<<(*this)[1]<<","<<(*this)[2]<<")"<<std::endl;
    }

    

};

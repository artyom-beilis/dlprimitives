///////////////////////////////////////////////////////////////////////////////
///
/// Copyright (c) 2021-2022 Artyom Beilis <artyomtnk@yahoo.com>
///
/// MIT License, see LICENSE.TXT
///
///////////////////////////////////////////////////////////////////////////////
#include <dlprim/json.hpp>
#include "test.hpp"
#include <iostream>
#include <limits>
#include <stdlib.h>
#include <sstream>
#include <iomanip>
using namespace dlprim;
using namespace std;

char const *jsn_str=
		"{ \"t\" : [{},{},{},{ \"x\":1},[]],\"x\" : { \"o\" : { \"test\" : [ 10,20,true ], \"post\" : 13.01 }, \"yes\" : \"\\u05d0א\" }}";

class parsing_error : public std::runtime_error {
public:
	parsing_error(std::string s) : std::runtime_error(s) {}
};

#define THROWS(X) do{ try { X; }catch(dlprim::json::bad_value_cast const &e){ break; }\
	catch(parsing_error const &e){ break; }\
	catch(...){\
	std::ostringstream tmp;	\
	tmp << __FILE__ << " " << __LINE__ << " "#X " not throwed correct error"; \
	throw std::runtime_error(tmp.str()); } \
	std::ostringstream tmp;	\
	tmp << __FILE__ << " " << __LINE__ << " "#X " not throwed at all"; \
	throw std::runtime_error(tmp.str()); \
}while(0)


json::value Parse(std::string s,int line)
{
	std::istringstream ss(s);
	json::value v2;
	char const *begin,*end;
	begin = s.c_str();
	end = begin + s.size();
	bool r=v2.load(begin,end,true);
	json::value v;
	if(!v.load(ss,true)) {
		TEST(!r);
		std::ostringstream tmp;
		tmp << "Parsing error of " << s << " in line " << line;
		throw parsing_error(tmp.str());
	}
	TEST(ss.get()==-1);
	TEST(begin == end);
	TEST(v==v2);
	return v;
}

std::string format(json::value const &v)
{
	std::ostringstream ss;
	ss<<v;
	return ss.str();
}

std::string deepa(int length)
{
    std::string a;
    for(int i=0;i<length;i++) 
        a += '[';

    for(int i=0;i<length;i++) 
        a += ']';
    return a;
}

std::string deepo(int length)
{
    std::string a;
    for(int i=0;i<length;i++) {
        a += "{\"x\":";
    }

    a+="1";

    for(int i=0;i<length;i++) 
        a += '}';
    return a;
}


#define parse(X) Parse(X,__LINE__)

int main()
{
	try {
		{
			dlprim::json::string_key a("a"),aa("a");
			dlprim::json::string_key b("b");
			TEST(a==aa);
			TEST(a!=b);
			TEST(a<b);
			TEST(a<=b);
			TEST(a<=a);
			TEST(b>a);
			TEST(b>=b);
			TEST(b>=a);
		}

		json::value v;
		json::value const &vc=v;
		TEST(v.type()==json::is_undefined);
		TEST(v.is_undefined());
		v=10;
		TEST(v.type()==json::is_number);
		TEST(v.number()==10);
		TEST(v.get_value<int>()==10);
		TEST(v.get_value<double>()==10);
		THROWS(v.get_value<std::string>());
		v="test";
		TEST(v.type()==json::is_string);
		TEST(v.str()=="test");
		v=true;
		TEST(v.type()==json::is_boolean);
		TEST(v.boolean()==true);
		v=json::null();
		TEST(v.is_null());
		TEST(v.type()==json::is_null);
		v=json::array();
		TEST(v.type()==json::is_array);
		v=json::object();
		TEST(v.type()==json::is_object);
		TEST(v.find("x")==json::value());
		TEST(v.type("x")==json::is_undefined);
		TEST(v.get<std::string>("x","y")=="y");
		THROWS(v.get<std::string>("x"));
		THROWS(v.at("x"));
		v["x"]=10;
		TEST(v.find("x")==json::value(10));
		TEST(v.at("x")==json::value(10));
		TEST(v.type("x")==json::is_number);
		TEST(v.get<std::string>("x","y")=="y");
		THROWS(v.get<std::string>("x"));
		v.set("x.y.z",10);
		TEST(v["x"]["y"]["z"].number()==10);
		TEST(v.get<int>("x.y.z")==10);
		TEST(parse("[]")==json::array());	
		TEST(parse("{}")==json::object());	
		TEST(parse("true")==json::value(true));	
		TEST(parse("false")==json::value(false));	
		TEST(parse("10")==json::value(10));	
		TEST(parse("\"hello\"")==json::value("hello"));	
		TEST(parse("null")==json::null());
        
        TEST(parse(deepa(100)).type() == json::is_array);
        TEST(parse(deepa(512)).type() == json::is_array);
        THROWS(parse(deepa(513)));
        THROWS(parse(deepa(51300)));

        TEST(parse(deepo(100)).type() == json::is_object);
        TEST(parse(deepo(512)).type() == json::is_object);
        THROWS(parse(deepo(513)));
        THROWS(parse(deepo(51300)));

		char const *s=
			"{ \"t\" : [{},{},{},{ \"x\":1},[]],\"x\" : { \"o\" : { \"test\" : [ 10,20,true ], \"post\" : 13.01 }, \"yes\" : \"\\u05d0א\" }}";
		v=parse(s);
		TEST(v.type("t")==json::is_array);
		TEST(v["t"].array().size()==5);
		TEST(v["t"][0]==json::object());
		TEST(v["t"][1]==json::object());
		TEST(v["t"][2]==json::object());
		TEST(v["t"][4]==json::array());
		TEST(v["t"][3]["x"].number()==1);
		TEST(v.type("x")==json::is_object);
		TEST(v.get<std::string>("x.yes")=="אא");
		// Test correct handing of surrogates
		THROWS(parse("\"\\ud834\""));
		THROWS(parse("\"\\udd1e\""));
		THROWS(parse("\"\\ud834 \\udd1e\""));
		TEST(parse("\"\\u05d0\\ud834\\udd1e x\"")=="א\xf0\x9d\x84\x9e x");
		THROWS(parse("\"\xFF\xFF\"")); // Invalid UTF-8
		TEST(parse("\"\\u05d0 x\"")=="א x"); // Correct read of 4 bytes
		THROWS(parse("\"\\u05dx x\"")); // Correct read of 4 bytes
		TEST(parse("[//Hello\n]")==json::array());
		TEST(format("test")=="\"test\"");
		TEST(format(10)=="10");
		TEST(format(true)=="true");
		TEST(format(false)=="false");
		TEST(format(json::null())=="null");
		TEST(format(json::object())=="{}");
		TEST(format(json::array())=="[]");
		v=json::value();
		v["x"]="yes";
		TEST(vc["x"].str()=="yes");
		THROWS(vc["y"]);
		THROWS(vc[1]);
		v[2]="yes";
		TEST(v[0]==json::null());
		TEST(v[1]==json::null());
		TEST(vc[2].str()=="yes");
		TEST(v[2]=="yes");
		TEST(v[3]==json::null());
		THROWS(vc[4]);
		THROWS(vc["x"]);
		v=json::value();
		v["x"]=10;
		v["y"][1]="test";
		v["y"][2]=json::array();
		TEST(format(v)=="{\"x\":10,\"y\":[null,\"test\",[]]}");

		std::string fl="123456789123456789123456789123456789";
		fl = fl.substr(0,std::numeric_limits<double>::digits10);
		double big_int=atof(fl.c_str());
		TEST(format(big_int)==fl);
		TEST(format(1277880000)=="1277880000");
		TEST(format(-1277880000)=="-1277880000");
		TEST(atoi(format(std::numeric_limits<int>::max()).c_str()) == std::numeric_limits<int>::max());
		TEST(atoi(format(std::numeric_limits<int>::min()).c_str()) == std::numeric_limits<int>::min());
		std::string fl2=fl.substr(0,1)+'.'+fl.substr(1);
		big_int=atof(fl2.c_str());
		TEST(format(big_int)==fl2);
		TEST(format(-big_int)=="-" + fl2);

		{
			std::string tmp;
			
			tmp = format(1.35e30);
			TEST(tmp.substr(0,5)=="1.35e");
			TEST(tmp.substr(tmp.size()-2)=="30");

			tmp = format(big_int * 1e30);
			TEST(tmp.substr(0,fl2.size()+1) == fl2+"e");
			TEST(tmp.substr(tmp.size()-2)=="30");
		}

		v["x"]=10000000;
		THROWS(v.get<short>("x"));
		TEST(v.get<int>("x")==10000000);
		v["x"]=-1;
		THROWS(v.get<unsigned>("x"));
		THROWS(v.get<unsigned short>("x"));
		THROWS(v.get<unsigned long >("x"));

		TEST(v.get<int>("x")==-1);
		TEST(v.get<short>("x")==-1);
		TEST(v.get<long>("x")==-1);

		/// FIXME
        //if(sizeof(long long) >= sizeof(double)) {
		//	THROWS(v["x"]=std::numeric_limits<long long>::max());
		//}
		
		if(std::numeric_limits<double>::max() != std::numeric_limits<float>::max()) {
			double val = std::numeric_limits<double>::max() / 100;
			v["x"]=val;
			THROWS(v.get<float>("x"));
			TEST(v.get<double>("x")==val);
			TEST(v.get<long double>("x")==val);
		}

		if(std::numeric_limits<long double>::max() != std::numeric_limits<double>::max()) {
			long double val = std::numeric_limits<long double>::max() / 100;
			THROWS(v["x"]=val);
		}
		char const *part="{}[]";
		TEST(v.load(part,part+4,false));
		TEST(*part=='[');
		TEST(v.type() == dlprim::json::is_object);
		TEST(!v.load(part,part+4,true));

		{
			json::value v;
			v["x"]=-10.0;
			TEST(v.get<float>("x")==-10.0f);
			TEST(v.get<double>("x")==-10.0);
			TEST(v.get<long double>("x")==-10.0);
		}

	}
	catch(std::exception const &e)
	{
		cerr<<"Failed:"<<e.what()<<endl;
		return 1;
	}
	std::cout << "Passed" << std::endl;
	return 0;
}


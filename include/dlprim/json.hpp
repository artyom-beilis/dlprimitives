#pragma once
///////////////////////////////////////////////////////////////////////////////
//                                                                             
//  Copyright (C) 2008-2022  Artyom Beilis (Tonkikh) <artyomtnk@yahoo.com>     
//                                                                             
// MIT License, see LICENSE.TXT
//
///////////////////////////////////////////////////////////////////////////////
#include <vector>
#include <map>
#include <string>
#include <iostream>
#include <limits>
#include <string.h>
#include <typeinfo>

#ifndef DLPRIM_API
#define DLPRIM_API
#endif

namespace dlprim {
///
/// \brief This namespace includes all JSON parsing and formatting related classes and functions
///
namespace json {
	///
	/// \brief a smart pointer similar to std::unique_ptr but it copies
	///   underlying object on pointer copy instead of moving its ownership.
	///
	/// Note: Underlying object has same constness as the pointer itself (not like in ordinary pointer).
	///
	/// Don't use it with polymorphic classes. Prefer clone_ptr instead.
	///
	template<typename T>
	class copy_ptr {
		T *ptr_;
	public:
		copy_ptr() : ptr_(0) {}
		explicit copy_ptr(T *v) : ptr_(v) {}
		copy_ptr(copy_ptr const &other) :
			ptr_(other.ptr_ ? new T(*other.ptr_) : 0)
		{
		}
		copy_ptr(copy_ptr &&other) : ptr_(other.ptr_)
		{
			other.ptr_ = 0;
		}
		copy_ptr &operator=(copy_ptr &&other) 
		{
			if(this!=&other) {
				this->swap(other);
				other.reset();
			}
			return *this;
		}
		copy_ptr const &operator=(copy_ptr const &other)
		{
			if(this != &other) {
				copy_ptr tmp(other);
				swap(tmp);
			}
			return *this;
		}
		~copy_ptr() {
			if(ptr_) delete ptr_;
		}

		T const *get() const { return ptr_; }
		T *get() { return ptr_; }

		T const &operator *() const { return *ptr_; }
		T &operator *() { return *ptr_; }
		T const *operator->() const { return ptr_; }
		T *operator->() { return ptr_; }
		T *release() { T *tmp=ptr_; ptr_=0; return tmp; }
		void reset(T *p=0)
		{
			if(ptr_) delete ptr_;
			ptr_=p;
		}
		void swap(copy_ptr &other)
		{
			T *tmp=other.ptr_;
			other.ptr_=ptr_;
			ptr_=tmp;
		}
	};
	///
	/// \brief This is a special object that may hold an std::string or
	/// alternatively reference to external (unowned) chunk of text
	///
	/// It is designed to be used for efficiency and reduce amount of
	/// memory allocations and copies.
	///
	/// It has interface that is roughly similar to the interface of std::string,
	/// but it does not provide a members that can mutate it or provide a NUL terminated
	/// string c_str().
	///
	class string_key {
	public:

		///
		/// Iterator type
		///
		typedef char const *const_iterator;

		///
		/// The last position of the character in the string
		///
		static const size_t npos = -1;
		
		///
		/// Default constructor - empty key
		///
		string_key() : 
			begin_(0),
			end_(0)
		{
		}

		///
		/// Create a new string copying the \a key
		///
		string_key(char const *key) :
			begin_(0),
			end_(0),
			key_(key)
		{
		}
		///
		/// Create a new string copying the \a key
		///
		string_key(std::string const &key) :
			begin_(0),
			end_(0),
			key_(key)
		{
		}
		///
		/// String size in bytes
		///
		size_t size() const
		{
			return end() - begin();
		}
		///
		/// Same as size()
		///
		size_t length() const
		{
			return size();
		}
		///
		/// Clear the string
		///
		void clear()
		{
			begin_ = end_ = 0;
			key_.clear();
		}
		///
		/// Check if the string is empty
		///
		bool empty() const
		{
			return end() == begin();
		}
		///
		/// Find first occurrence of a character \c in the string starting from
		/// position \a pos. Returns npos if not character found.
		///
		size_t find(char c,size_t pos = 0) const
		{
			size_t s = size();
			if(pos >= s)
				return npos;
			char const *p=begin() + pos;
			while(pos <= s && *p!=c) {
				pos++;
				p++;
			}
			if(pos >= s)
				return npos;
			return pos;
		}

		///
		/// Create a substring from this string starting from character \a pos of size at most \a n
		///
		string_key substr(size_t pos = 0,size_t n=npos) const
		{
			string_key tmp = unowned_substr(pos,n);
			return string_key(std::string(tmp.begin(),tmp.end()));
		}
		///
		/// Create a substring from this string starting from character \a pos of size at most \a n
		/// such that the memory is not copied but only reference by the created substring
		///
		string_key unowned_substr(size_t pos = 0,size_t n=npos) const
		{
			if(pos >= size()) {
				return string_key();
			}
			char const *p=begin() + pos;
			char const *e=end();
			if(n > size_t(e-p)) {
				return string_key(p,e);
			}
			else {
				return string_key(p,p+n);
			}
		}
		
		///
		/// Get a character at position \a n
		///
		char const &operator[](size_t n) const
		{
			return *(begin() + n);
		}
		///
		/// Get a character at position \a n, if \a n is not valid position, throws std::out_of_range exception
		///
		char const &at(size_t n) const
		{
			if(n > size())
				throw std::out_of_range("dlprim::string_key::at() range error");
			return *(begin() + n);
		}

		///
		/// Create a string from \a v without copying the memory. \a v should remain valid
		/// as long as this object is used
		///	
		static string_key unowned(std::string const &v) 
		{
			return string_key(v.c_str(),v.c_str()+v.size());
		}
		///
		/// Create a string from \a str without copying the memory. \a str should remain valid
		/// as long as this object is used
		///	
		static string_key unowned(char const *str) 
		{
			char const *end = str;
			while(*end)
				end++;
			return string_key(str,end);
		}
		///
		/// Create a string from \a characters at rang [begin,end) without copying the memory.
		/// The range should remain valid as long as this object is used
		///	
		static string_key unowned(char const *begin,char const *end) 
		{
			return string_key(begin,end);
		}

		///
		/// Get a pointer to the first character in the string
		///
		char const *begin() const
		{
			if(begin_)
				return begin_;
			return key_.c_str();
		}
		///
		/// Get a pointer to the one past last character in the string
		///
		char const *end() const
		{
			if(begin_)
				return end_;
			return key_.c_str() + key_.size();
		}
		///
		/// Compare two strings
		///
		bool operator<(string_key const &other) const
		{
			return std::lexicographical_compare(	begin(),end(),
								other.begin(),other.end(),
								std::char_traits<char>::lt);
		}
		///
		/// Compare two strings
		///
		bool operator>(string_key const &other) const
		{
			return other < *this;
		}
		///
		/// Compare two strings
		///
		bool operator>=(string_key const &other) const
		{
			return !(*this < other);
		}
		///
		/// Compare two strings
		///
		bool operator<=(string_key const &other) const
		{
			return !(*this > other);
		}
		///
		/// Compare two strings
		///
		bool operator==(string_key const &other) const
		{
			return (end() - begin() == other.end() - other.begin())
				&& memcmp(begin(),other.begin(),end()-begin()) == 0;
		}
		///
		/// Compare two strings
		///
		bool operator!=(string_key const &other) const
		{
			return !(*this==other);
		}

		///
		/// Get the pointer to the first character in the string. Note it should not be NUL terminated
		///
		char const *data() const
		{
			return begin();
		}

		///
		/// Create std::string from the key
		///
		std::string str() const
		{
			if(begin_)
				return std::string(begin_,end_-begin_);
			else
				return key_;
		}
		///
		/// Convert the key to the std::string
		///
		operator std::string() const
		{
			return str();
		}
	private:
		string_key(char const *b,char const *e) :
			begin_(b),
			end_(e)
		{
		}

		char const *begin_;
		char const *end_;
		std::string key_;
	};

	///
	/// Write the string to the stream
	///
	inline std::ostream &operator<<(std::ostream &out,string_key const &s)
	{
		out.write(s.data(),s.size());
		return out;
	}

	///
	/// Compare two strings
	///
	inline bool operator==(string_key const &l,char const *r)
	{
		return l==string_key::unowned(r);
	}

	///
	/// Compare two strings
	///
	inline bool operator==(char const *l,string_key const &r)
	{
		return string_key::unowned(l) == r;
	}

	///
	/// Compare two strings
	///
	inline bool operator==(string_key const &l,std::string const &r)
	{
		return l==string_key::unowned(r);
	}


	///
	/// Compare two strings
	///
	inline bool operator==(std::string const &l,string_key const &r)
	{
		return string_key::unowned(l) == r;
	}

	///
	/// Compare two strings
	///
	inline bool operator!=(string_key const &l,char const *r)
	{
		return l!=string_key::unowned(r);
	}

	///
	/// Compare two strings
	///
	inline bool operator!=(char const *l,string_key const &r)
	{
		return string_key::unowned(l) != r;
	}

	///
	/// Compare two strings
	///
	inline bool operator!=(string_key const &l,std::string const &r)
	{
		return l!=string_key::unowned(r);
	}

	///
	/// Compare two strings
	///
	inline bool operator!=(std::string const &l,string_key const &r)
	{
		return string_key::unowned(l) != r;
	}
	///
	/// Compare two strings
	///
	inline bool operator<=(string_key const &l,char const *r)
	{
		return l<=string_key::unowned(r);
	}

	///
	/// Compare two strings
	///
	inline bool operator<=(char const *l,string_key const &r)
	{
		return string_key::unowned(l) <= r;
	}

	///
	/// Compare two strings
	///
	inline bool operator<=(string_key const &l,std::string const &r)
	{
		return l<=string_key::unowned(r);
	}

	///
	/// Compare two strings
	///
	inline bool operator<=(std::string const &l,string_key const &r)
	{
		return string_key::unowned(l) <= r;
	}
	///
	/// Compare two strings
	///
	inline bool operator>=(string_key const &l,char const *r)
	{
		return l>=string_key::unowned(r);
	}

	///
	/// Compare two strings
	///
	inline bool operator>=(char const *l,string_key const &r)
	{
		return string_key::unowned(l) >= r;
	}

	///
	/// Compare two strings
	///
	inline bool operator>=(string_key const &l,std::string const &r)
	{
		return l>=string_key::unowned(r);
	}

	///
	/// Compare two strings
	///
	inline bool operator>=(std::string const &l,string_key const &r)
	{
		return string_key::unowned(l) >= r;
	}


	///
	/// Compare two strings
	///
	inline bool operator<(string_key const &l,char const *r)
	{
		return l<string_key::unowned(r);
	}

	///
	/// Compare two strings
	///
	inline bool operator<(char const *l,string_key const &r)
	{
		return string_key::unowned(l) < r;
	}

	///
	/// Compare two strings
	///
	inline bool operator<(string_key const &l,std::string const &r)
	{
		return l<string_key::unowned(r);
	}

	///
	/// Compare two strings
	///
	inline bool operator<(std::string const &l,string_key const &r)
	{
		return string_key::unowned(l) < r;
	}
	///
	/// Compare two strings
	///
	inline bool operator>(string_key const &l,char const *r)
	{
		return l>string_key::unowned(r);
	}

	///
	/// Compare two strings
	///
	inline bool operator>(char const *l,string_key const &r)
	{
		return string_key::unowned(l) > r;
	}

	///
	/// Compare two strings
	///
	inline bool operator>(string_key const &l,std::string const &r)
	{
		return l>string_key::unowned(r);
	}

	///
	/// Compare two strings
	///
	inline bool operator>(std::string const &l,string_key const &r)
	{
		return string_key::unowned(l) > r;
	}


	class value;

	///
	/// \brief Special object that is convertible to null json value
	///
	struct null {};
	///
	/// \brief Special object that is convertible to undefined json value
	///
	struct undefined {};

	inline bool operator==(undefined const &/*l*/,undefined const &/*r*/) {return true;}
	inline bool operator!=(undefined const &/*l*/,undefined const &/*r*/) {return false;}
	inline bool operator==(null const &/*l*/,null const &/*r*/) {return true;}
	inline bool operator!=(null const &/*l*/,null const &/*r*/) {return false;}

	///
	/// \brief The json::array - std::vector of json::value's
	///
	typedef std::vector<value> array;
	///
	/// \brief The json::object - std::map of json::value's
	///
	typedef std::map<string_key,value> object;

	
	template<typename T>
	struct traits;


	///
	/// The type of json value
	///
	typedef enum {
		is_undefined,	///< Undefined value
		is_null,	///< null value
		is_boolean,	///< boolean value
		is_number,	///< numeric value
		is_string,	///< string value
		is_object,	///< object value 
		is_array	///< array value
	} json_type;


	enum {
		compact = 0, ///< Print JSON values in most compact format
		readable = 1 ///< Print JSON values in human readable format (with identention)
	};
	
	///
	/// \brief The error that is thrown in case of bad conversion of json::value to ordinary value
	///
	/// When implementing json::traits for complex classes you are expected to throw this exception
	/// in case of invalid formatting
	///
	class DLPRIM_API bad_value_cast : public std::bad_cast {
	public:
		bad_value_cast(); 
		bad_value_cast(std::string const &s);
		bad_value_cast(std::string const &s,json_type actual);
		bad_value_cast(std::string const &s,json_type expected, json_type actual);

		virtual ~bad_value_cast() throw();
		virtual const char* what() const throw();
	private:
		std::string msg_;
	};
	
	class value;

	///
	/// Read json object from input stream
	///
	std::istream DLPRIM_API &operator>>(std::istream &in,value &v);

	///
	/// Write json object to output stream
	///
	std::ostream DLPRIM_API &operator<<(std::ostream &out,value const &v);

	///
	/// Write human readable representation of json_type 
	///
	std::ostream DLPRIM_API &operator<<(std::ostream &out,json_type);

	///
	/// \brief This class is central representation of json objects.
	///
	/// It can be a value from any type
	/// including scalar, object, array and undefined
	///
	class DLPRIM_API value {
	public:

		///
		/// Get the type of the value
		/// 
		json_type type() const;
		
		///
		/// Returns true if type()==json::is_undefined
		///
		bool is_undefined() const;
		///
		/// Returns true if type()==json::is_null
		///
		bool is_null() const;

		///
		/// Convert value to boolean, throws bad_value_cast if value's type is not boolean
		///
		bool const &boolean() const;
		///
		/// Convert value to double, throws bad_value_cast if value's type is not number
		///
		double const &number() const;
		///
		/// Convert value to strng, throws bad_value_cast if value's type is not string
		///
		std::string const &str() const;
		///
		/// Convert value to json::object, throws bad_value_cast if value's type is not object
		///
		json::object const &object() const;
		///
		/// Convert value to json::array, throws bad_value_cast if value's type is not array
		///
		json::array const &array() const;

		///
		/// Get reference to bool variable that represents this value, throws bad_value_cast if type is invalid
		///
		bool &boolean();
		///
		/// Get reference to double variable that represents this value, throws bad_value_cast if type is invalid
		///
		double &number();
		///
		/// Get reference to string variable that represents this value, throws bad_value_cast if type is invalid
		///
		std::string &str();
		///
		/// Get reference to object variable that represents this value, throws bad_value_cast if type is invalid
		///
 		json::object &object();
		///
		/// Get reference to array variable that represents this value, throws bad_value_cast if type is invalid
		///
		json::array &array();

		///
		/// Set value to undefined type
		///
		void undefined();
		///
		/// Set value to null type
		///
		void null();

		///
		/// Set value to boolean type and assign it
		///
		void boolean(bool);
		///
		/// Set value to numeric type and assign it
		///
		void number(double );
		///
		/// Set value to string type and assign it
		///
		void str(std::string const &);
		///
		/// Set value to object type and assign it
		///
		void object(json::object const &);
		///
		/// Set value to array type and assign it
		///
		void array(json::array const &);


		///
		/// Convert the value to type T, using json::traits, throws bad_value_cast if conversion is not possible
		///
		template<typename T>
		T get_value() const
		{
			return traits<T>::get(*this);
		}
		
		///
		/// Convert the object \a v of type T to the value
		/// 
		template<typename T>
		void set_value(T const &v)
		{
			traits<T>::set(*this,v);
		}

		///
		/// Searches a value in the path \a path
		///
		/// For example if the json::value represents { "x" : { "y" : 10 } }, then find("x.y") would return
		/// a reference to value that hold a number 10, find("x") returns a reference to a value
		/// that holds an object { "y" : 10 } and find("foo") would return value of undefined type.
		///
		value const &find(std::string const &path) const; 		
		///
		/// Searches a value in the path \a path
		///
		/// For example if the json::value represents { "x" : { "y" : 10 } }, then find("x.y") would return
		/// a reference to value that hold a number 10, find("x") returns a reference to a value
		/// that holds an object { "y" : 10 } and find("foo") would return value of undefined type.
		///
		value const &find(char const *path) const; 		

		///
		/// Searches a value in the path \a path, if not found throw bad_value_cast.
		///
		/// For example if the json::value represents { "x" : { "y" : 10 } }, then find("x.y") would return
		/// a reference to value that hold a number 10, find("x") returns a reference to a value
		/// that holds an object { "y" : 10 } and find("foo") throws
		///
		value const &at(std::string const &path) const;  
		///
		/// Searches a value in the path \a path, if not found throw bad_value_cast.
		///
		/// For example if the json::value represents { "x" : { "y" : 10 } }, then find("x.y") would return
		/// a reference to value that hold a number 10, find("x") returns a reference to a value
		/// that holds an object { "y" : 10 } and find("foo") throws
		///
		value const &at(char const *path) const;  
		///
		/// Searches a value in the path \a path, if not found throw bad_value_cast.
		///
		/// For example if the json::value represents { "x" : { "y" : 10 } }, then find("x.y") would return
		/// a reference to value that hold a number 10, find("x") returns a reference to a value
		/// that holds an object { "y" : 10 } and find("foo") throws
		///
		value &at(std::string const &path);
		///
		/// Searches a value in the path \a path, if not found throw bad_value_cast.
		///
		/// For example if the json::value represents { "x" : { "y" : 10 } }, then find("x.y") would return
		/// a reference to value that hold a number 10, find("x") returns a reference to a value
		/// that holds an object { "y" : 10 } and find("foo") throws
		///
		value &at(char const *path);

		///
		/// Sets the value \a v at the path \a path, if the path invalid, creates it.
		///
		void at(std::string const &path,value const &v);
		///
		/// Sets the value \a v at the path \a path, if the path invalid, creates it.
		///
		void at(char const *path,value const &v);

		
		///
		/// Creates a value from and object \a v of type T
		///
		template<typename T>
		value(T const &v)
		{
			set_value(v);
		}

		///
		/// Returns the type of variable in path, if not found returns undefined
		///
		/// Same as find(path).type()
		///
		json_type type(std::string const &path) const
		{
			return find(path).type();
		}
		///
		/// Returns the type of variable in path, if not found returns undefined
		///
		/// Same as find(path).type()
		///
		json_type type(char const *path) const
		{
			return find(path).type();
		}

		///
		/// Convert an object \a v of type T to a value at specific path, same as at(path,value(v))
		///
		template<typename T>
		void set(std::string const &path,T const &v)
		{
			at(path,value(v));
		}
		///
		/// Convert an object \a v of type T to a value at specific path, same as at(path,value(v))
		///
		template<typename T>
		void set(char const *path,T const &v)
		{
			at(path,value(v));
		}

		///
		/// Get a string value from a path \a path. If the path is not invalid or the object
		/// is not of type string at this path, returns \a def instead
		///
		std::string get(std::string const &path,char const *def) const
		{
			value const &v=find(path);
			if(v.is_undefined())
				return def;
			try {
				return v.get_value<std::string>();
			}
			catch(std::bad_cast const &e) {
				return def;
			}
		}
		///
		/// Get a string value from a path \a path. If the path is not invalid or the object
		/// is not of type string at this path, returns \a def instead
		///
		std::string get(char const *path,char const *def) const
		{
			value const &v=find(path);
			if(v.is_undefined())
				return def;
			try {
				return v.get_value<std::string>();
			}
			catch(std::bad_cast const &e) {
				return def;
			}
		}
		
		///
		/// Get an object of type T from the path \a path. Throws bad_value_cast if such path does not
		/// exists of conversion can't be done
		///
		template<typename T>
		T get(std::string const &path) const
		{
			return at(path).get_value<T>();
		}
		///
		/// Get an object of type T from the path \a path. Throws bad_value_cast if such path does not
		/// exists of conversion can't be done
		///
		template<typename T>
		T get(char const *path) const
		{
			return at(path).get_value<T>();
		}

		///
		/// Get an object of type T from the path \a path. Returns \a def if such path does not
		/// exists of conversion can't be done
		///
		template<typename T>
		T get(char const *path,T const &def) const
		{
			value const &v=find(path);
			if(v.is_undefined())
				return def;
			try {
				return v.get_value<T>();
			}
			catch(std::bad_cast const &e) {
				return def;
			}
		}
		///
		/// Get an object of type T from the path \a path. Returns \a def if such path does not
		/// exists of conversion can't be done
		///
		template<typename T>
		T get(std::string const &path,T const &def) const
		{
			value const &v=find(path);
			if(v.is_undefined())
				return def;
			try {
				return v.get_value<T>();
			}
			catch(std::bad_cast const &e) {
				return def;
			}
		}

		///
		/// Returns a reference to the node \a name of the value. 
		/// For value = {"x",10} and name == "x" return a value that holds 10.
		///
		/// If value is not object it's type set to object.
		/// If such node does not exits, it is created with undefined value
		///
		value &operator[](std::string const &name);

		///
		/// Returns reference to the node \a name of the value.
		/// For value = {"x",10} and name == "x" return a value that holds 10.
		///
		/// If value is not object or such node does not exits, throws bad_value_cast
		///
		value const &operator[](std::string const &name) const;

		///
		/// Returns a reference to \a n 'th entry of the array. If the value is not an array it is reset to array,
		/// of the array is too small it is resized to size of at least n+1
		///
		value &operator[](size_t n);
		///
		/// Returns a reference to \a n 'th entry of array, if the value is not array or n is too big, throws
		/// bad_value_cast
		///
		value const &operator[](size_t n) const;

		///
		/// Convert a value to std::string, if \a how has value \a readable it is converted with indentation
		///
		std::string save(int how=compact) const;
		///
		/// Write a value to std::ostream, if \a how has value \a readable it is converted with indentation
		///
		void save(std::ostream &out,int how=compact) const;
		///
		/// Read a value from std::istream.
		///
		/// Note: only JSON object and JSON array are considered valid values
		///
		/// \param in the std::istream used to read the data
		/// \param full  require EOF once the object is read, otherwise consider it as syntax error
		/// \param line_number  return a number of the line where syntax error occurred
		/// \result returns true if the value was read successfully, otherwise returns false to indicate a syntax error.
		///
		bool load(std::istream &in,bool full,int *line_number=0);

		///
		/// Read a value from character range
		///
		/// Note: only JSON object and JSON array are considered valid values
		///
		/// \param begin - begin of the text range, at the end points to the end of parsed range
		/// \param end - end of the text range
		/// \param full  require EOF once the object is read, otherwise consider it as syntax error
		/// \param line_number  return a number of the line where syntax error occurred
		/// \result returns true if the value was read successfully, otherwise returns false to indicate a syntax error.
		///
		bool load(char const *&begin,char const *end,bool full,int *line_number=0);

		///
		/// Compare two values objects, return true if they are same
		///
		bool operator==(value const &other) const;
		///
		/// Compare two values objects, return false if they are same
		///
		bool operator!=(value const &other) const;


		///
		/// Move assignment
		///
		value &operator=(value &&other)
		{
			d=std::move(other.d);
			return *this;
		}
		///
		/// Move constructor
		///
		value(value &&other) : d(std::move(other.d))
		{
			
		}
		///
		/// Copy constructor
		///
		value(value const &other) :
			d(other.d)
		{
		}
		///
		/// Assignment operator
		///
		value const &operator=(value const &other)
		{
			d=other.d;
			return *this;
		}
		///
		/// Default value - creates a value of undefined type
		///
		value()
		{
		}
		
		///
		/// Destructor
		///

		~value()
		{
		}

		///
		/// Swaps two values, does not throw.
		///
		void swap(value &other)
		{
			d.swap(other.d);
		}

	private:

		void write(std::ostream &out,int tabs) const;
		void write_value(std::ostream &out,int tabs) const;

		struct _data;
		struct DLPRIM_API copyable {

			_data *operator->() { return &*d; }
			_data &operator*() { return *d; }
			_data const *operator->() const { return &*d; }
			_data const &operator*() const { return *d; }

			copyable();
			copyable(copyable const &r);
			copyable(copyable &&);
			copyable &operator=(copyable &&r);
			copyable const &operator=(copyable const &r);
			~copyable();

			void swap(copyable &other) 
			{
				d.swap(other.d);
			}
		private:
			copy_ptr<_data> d;
		} d;

		friend struct copyable;
	};


	///
	/// Convert UTF-8 string to JSON string, i.e. a sring foo is converted to "foo",
	/// a string bar"baz is converted to "bar\"baz"
	///
	std::string DLPRIM_API to_json(std::string const &utf);
	///
	/// Convert UTF-8 string to JSON string, i.e. a sring foo is converted to "foo",
	/// a string bar"baz is converted to "bar\"baz"
	///
	std::string DLPRIM_API to_json(char const *begin,char const *end);
	///
	/// Convert UTF-8 string to JSON string, i.e. a sring foo is converted to "foo",
	/// a string bar"baz is converted to "bar\"baz"
	///
	void DLPRIM_API to_json(char const *begin,char const *end,std::ostream &out);
	//
	/// Convert UTF-8 string to JSON string, i.e. a sring foo is converted to "foo",
	/// a string bar"baz is converted to "bar\"baz"
	///
	void DLPRIM_API to_json(std::string const &str,std::ostream &out);

	
	/// \cond INTERNAL

	template<typename T1,typename T2>
	struct traits<std::pair<T1,T2> > {
		static std::pair<T1,T2> get(value const &v)
		{
			if(v.object().size()!=2)
				throw bad_value_cast("Object with two members expected");
			std::pair<T1,T2> pair(v.get_value<T1>("first"),v.get_value<T2>("second"));
			return pair;
		}
		static void set(value &v,std::pair<T1,T2> const &in)
		{
			v=json::object();
			v.set_value("first",in.first);
			v.set_value("second",in.second);
		}
	};

	template<typename T>
	struct traits<std::vector<T> > {
		static std::vector<T> get(value const &v)
		{
			std::vector<T> result;
			json::array const &a=v.array();
			result.resize(a.size());
			for(unsigned i=0;i<a.size();i++) 
				result[i]=a[i].get_value<T>();
			return result;
		}
		static void set(value &v,std::vector<T> const &in)
		{
			v=json::array();
			json::array &a=v.array();
			a.resize(in.size());
			for(unsigned i=0;i<in.size();i++)
				a[i].set_value(in[i]);
		}
	};


	#define DLPRIM_JSON_SPECIALIZE(type,method) 	\
	template<>					\
	struct traits<type> {				\
		static type get(value const &v)		\
		{					\
			return v.method();		\
		}					\
		static void set(value &v,type const &in)\
		{					\
			v.method(in);			\
		}					\
	};

	DLPRIM_JSON_SPECIALIZE(bool,boolean);
	DLPRIM_JSON_SPECIALIZE(double,number);
	DLPRIM_JSON_SPECIALIZE(std::string,str);
	DLPRIM_JSON_SPECIALIZE(json::object,object);
	DLPRIM_JSON_SPECIALIZE(json::array,array);

	#undef DLPRIM_JSON_SPECIALIZE
	
	#define DLPRIM_JSON_SPECIALIZE_INT(type) 			\
	template<>							\
	struct traits<type> {						\
		static type get(value const &v)				\
		{							\
			type res=static_cast<type>(v.number());		\
			if(res!=v.number())				\
				throw bad_value_cast();			\
			return res;					\
		}							\
		static void set(value &v,type const &in)		\
		{							\
			if(std::numeric_limits<type>::digits >		\
				std::numeric_limits<double>::digits	\
				&& static_cast<double>(in)!=in)		\
			{						\
				throw bad_value_cast();			\
			}						\
			v.number(static_cast<double>(in));		\
		}							\
	};

	DLPRIM_JSON_SPECIALIZE_INT(char)
	DLPRIM_JSON_SPECIALIZE_INT(unsigned char)
	DLPRIM_JSON_SPECIALIZE_INT(signed char)
	DLPRIM_JSON_SPECIALIZE_INT(wchar_t)
	DLPRIM_JSON_SPECIALIZE_INT(short)
	DLPRIM_JSON_SPECIALIZE_INT(unsigned short)
	DLPRIM_JSON_SPECIALIZE_INT(int)
	DLPRIM_JSON_SPECIALIZE_INT(unsigned int)
	DLPRIM_JSON_SPECIALIZE_INT(long)
	DLPRIM_JSON_SPECIALIZE_INT(unsigned long)
	DLPRIM_JSON_SPECIALIZE_INT(long long)
	DLPRIM_JSON_SPECIALIZE_INT(unsigned long long)

	#undef DLPRIM_JSON_SPECIALIZE_INT

	template<>
	struct traits<float> {
		static float get(value const &v)
		{
			double r=v.number();
			if(	r < (-std::numeric_limits<float>::max()) // actually should be C++11 lowest, but it should be under IEEE float lowest()=-max()
			     || std::numeric_limits<float>::max() < r )
			{
				throw bad_value_cast();
			}
			return static_cast<float>(r);
		}
		static void set(value &v,float const &in)
		{
			v.number(in);
		}
	};
	
	template<>
	struct traits<long double> {
		static long double get(value const &v)
		{
			return v.number();
		}
		static void set(value &v,long double const &in)
		{
			if( in < -std::numeric_limits<double>::max() // should actually be std::numeric_limits<float>::lowest() but it is ==-max()
			     || std::numeric_limits<double>::max() < in )
			{
				throw bad_value_cast();
			}
			v.number(static_cast<double>(in));
		}
	};

	template<>					
	struct traits<json::null> {				
		static void set(value &v,json::null const &/*in*/)
		{					
			v.null();
		}					
	};

	template<int n>					
	struct traits<char[n]> {			
		typedef char vtype[n];
		static void set(value &v,vtype const &in)
		{					
			v.str(in);
		}					
	};
	template<int n>					
	struct traits<char const [n]> {			
		typedef char const vtype[n];
		static void set(value &v,vtype const &in)
		{					
			v.str(in);
		}
	};

	
	template<>					
	struct traits<char const *> {			
		static void set(value &v,char const * const &in)
		{					
			v.str(in);
		}					
	};
	
	/// \endcond


} // json
} // dlprim


#!/usr/bin/python
# -*- coding: utf-8 -*-
#

# Requires bitarray and mmh3 (murmurhash3)
# sudo pip install bitarray
# sudo pip install mmh3

from bitarray import bitarray
import mmh3
from math import log, ceil

class BloomFilter:

	def __init__(self, capacity, error_rate):
		"""
		A Bloom Filter is a space-efficient probabilistic data structure.
		This class implements a classic Bloom Filter.

		capacity
			The number of elements this Bloom Filter must be able to store 
			with the specfied maximum error_rate.
		error_rate
			The maximum probability of false positives allowed for the specified capacity.
		"""
		if not capacity > 0: raise ValueError("capacity must be > 0")
		if not (0 < error_rate < 1): raise ValueError("error_rate must be between 0 and 1.")
		# The expected maximum capacity. The number of stored elements may exceed this capacity.
		self.capacity = capacity
		# The number of stored elements.
		self.element_count = 0
		# The maximum probability of false positives allowed for the specified capacity.
		self.error_rate = error_rate
		# Size of the bit array.
		self.size = self.calc_size()
		# The number of hashes used to store/check an element.
		self.hash_count = self.calc_hash_count()
		# The bit array.
		self.bit_array = bitarray(self.size)
		self.bit_array.setall(0)

	def add(self, element):
		"""
		Adds an element to this Bloom Filter.
		This method doesn't check the capacity.
		An item count higher than capacity makes the error rate higher than expected.

		element
			Element to add. Must be a string.
		"""
		for seed in xrange(self.hash_count):
			position = mmh3.hash(element, seed) % self.size
			self.bit_array[position] = 1
		self.element_count += 1

	def lookup(self, element):
		"""
		Tests an element's membership in this Bloom Filter.
		False positives are possible, but false negatives are not.

		A 'True' value returned means 'possibly in set'.
		A 'False' value returned means 'definitely not in set'.

		element
			Element to test. Must be a string.
		"""
		for seed in xrange(self.hash_count):
			position = mmh3.hash(element, seed) % self.size
			if self.bit_array[position] == 0:
				return False
		return True

	def union(self, b):
		"""
		Returns the result of the union operation between this Bloom Filter and another one.
		The union operation is lossless: the result is be the same as the Bloom Filter 
		created from scratch using the union of the two sets of elements.

		To be able to perform the union operation, the size and hash count of both filters 
		must be the same.

		The result is a new Bloom Filter with the same capacity and error_rate as this 
		Bloom Filter (so the error rate may be different than the error rate of the second 
		Bloom Filter used in the operation). Note that the total element count may exceed the 
		capacity, so the real error rate may also be higher.

		b
			Second Bloom Filter for the operation.
		"""
		if self.size != b.size: return None
		if self.hash_count != b.hash_count: return None
		result = BloomFilter(self.capacity, self.error_rate)
		result.bit_array = self.bit_array | b.bit_array
		result.element_count = self.element_count + b.element_count
		return result

	def intersection(self, b):
		"""
		Returns the result of the intersection operation between this Bloom Filter and 
		another one.
		The result may not be as accurate as a Bloom Filter created from scratch with the 
		intersection of the two sets of elements (wrong number of elements, higher error rate).

		To be able to perform the intersection operation, the size and hash count of both 
		filters must be the same.

		The result is a new Bloom Filter with the same capacity and error_rate as this 
		Bloom Filter (so the error rate may be different than the error rate of the second 
		Bloom Filter used in the operation). Note that the element count of the returned filter 
		is just an estimation.

		b
			Second Bloom Filter for the operation.
		"""
		if self.size != b.size: return None
		if self.hash_count != b.hash_count: return None
		result = BloomFilter(self.capacity, self.error_rate)
		result.bit_array = self.bit_array & b.bit_array
		result.element_count = result.calc_element_count()
		return result

	def is_full(self):
		"""
		Returns True if the filter is at full capacity.
		"""
		if self.element_count < self.capacity: return False
		else: return True

	def calc_size(self):
		"""
		Calculates the size for the bit array, 
		based on the given capacity and error_rate.
		"""
		# m = - (n * ln(p)) / (ln(2))**2
		return int(ceil(- (float(self.capacity) * log(float(self.error_rate))) / (log(2))**2))

	def calc_hash_count(self):
		"""
		Calculates the number of hashes needed to store each element, 
		based on the given capacity and the size of the bit array.
		"""
		# k = (m/n) * ln(2)
		return int(ceil((float(self.size) / float(self.capacity)) * log(2)))

	def calc_error_rate(self, use_capacity = False):
		"""
		Calculates the current (real) error rate of the filter.
		The error rate depends on the size of the bit array, the number 
		of hashes and the number of elements stored in the filter.

		use_capacity
			If set to True, the capacity will be used instead of the current 
			element_count. The result will be the estimated error rate, wich 
			should be the same as the error_rate set at initialization.
		"""
		# p = (1 - (1 - 1/m)**(k*n))**k
		if use_capacity: n = float(self.capacity)
		else: n = float(self.element_count)
		return (1.0 - (1.0 - 1.0 / float(self.size)) ** (float(self.hash_count) * n)) ** float(self.hash_count)

	def calc_element_count(self):
		"""
		Calculates an estimation of the number of elements currently 
		stored in the filter.

		The estimation is based on the size of the bit array, the number 
		of bits set to 1 and the number of hashes used.

		The element_count attribute should be used instead of this method 
		if possible.
		"""
		# i = - (m * ln(1 - (x/m))) / k
		x = float(self.bit_array.count())
		return int(ceil(- (float(self.size) * log(1.0 - (x / float(self.size)))) / float(self.hash_count)))

	def __contains__(self, string):
		return self.lookup(string)

class ScalableBloomFilter:

	SCALE_MODE_LINEAR = 1
	SCALE_MODE_EXPONENTIAL = 2

	def __init__(self, initial_capacity, error_rate, scale_factor = 2, scale_mode = SCALE_MODE_LINEAR):
		"""
		A Scalable Bloom Filter is a Bloom Filter that grows every time 
		it reaches the maximum capacity.
		This class implements a Scalable Bloom Filter with some flexible 
		scaling options.

		The growth is implemented by creating new Bloom Filters with more 
		capacity. New elements are always added to the last filter, keeping 
		the previous ones at full capacity.

		initial_capacity
			The capacity for the first Bloom Filter.
		error_rate
			The error_rate for this Bloom Filter.
		scale_factor
			Determines how quickly the filter grows. Must be higher than 0.
		scale_mode
			Determines if the growth must be linear or exponential.
		"""
		if not initial_capacity > 0: raise ValueError("initial_capacity must be > 0.")
		if not (0 < error_rate < 1): raise ValueError("error_rate must be between 0 and 1.")
		if not scale_factor > 0: raise ValueError("scale_factor must be > 0.")
		if not scale_mode in [1, 2]: raise ValueError("Invalid scale_mode, use one of the SCALE_MODE_* constants.")
		# The capacity of the first filter.
		self.initial_capacity = initial_capacity
		# Total number of elements added.
		self.element_count = 0
		# The error rate shared by all filters.
		self.error_rate = error_rate
		# The scaling factor.
		self.scale_factor = scale_factor
		# The scaling mode.
		self.scale_mode = scale_mode
		# The filters array.
		self.filters = []
		self.filters.append(BloomFilter(self.initial_capacity, self.error_rate))

	def add(self, element):
		"""
		Adds an element to this Bloom Filter.
		If the filter currently used is at full capacity, a new one is created.

		element
			Element to add. Must be a string.
		"""
		if self.filters[-1].is_full():
			self.filters.append(BloomFilter(self.calc_next_capacity(), self.error_rate))
		self.filters[-1].add(element)
		self.element_count += 1

	def lookup(self, string):
		"""
		Tests an element's membership in this Bloom Filter.
		False positives are possible, but false negatives are not.

		A 'True' value returned means 'possibly in set'.
		A 'False' value returned means 'definitely not in set'.

		element
			Element to test. Must be a string.
		"""
		for f in reversed(self.filters):
			if f.lookup(string): return True
		return False

	def calc_next_capacity(self):
		"""
		Calculates the capacity that the next filter should have according 
		to the scaling mode and factor.
		"""
		if self.scale_mode == self.SCALE_MODE_LINEAR:
			capacity = int(self.initial_capacity * (self.scale_factor * len(self.filters)))
		else: capacity = int(self.initial_capacity * (self.scale_factor ** len(self.filters)))
		return capacity

	def union(self, b):
		"""
		Returns the result of the union operation between this Scalable 
		Bloom Filter and another one.

		The two filters may have different initial capacities, error rates 
		and scaling options.

		The result is a new Scalable Bloom Filter with the parameters of 
		this Scalable Bloom Filter (same initial_capacity, error_rate, 
		scale_factor and scale_mode) but the filters array will not be 
		consistent with these parameters as it will contain the sum of the 
		two filters arrays.

		Note that, if the second filter has a different error rate than this 
		filter, the error rate of the resulting filter will not be real.

		Also note that the resulting filter array may have 2 filters not 
		full (the last filters of both arrays) and adding new elements will 
		only fill the second one, leaving the first one untouched. This 
		may be highly inefficient if that filter is very large and has 
		only a few elements stored.

		b
			Second Scalable Bloom Filter for the operation.
		"""
		result = ScalableBloomFilter(self.initial_capacity, self.error_rate, self.scale_factor, self.scale_mode)
		result.filters = self.filters + b.filters
		result.element_count = self.element_count + b.element_count
		return result

	def __contains__(self, string):
		return self.lookup(string)

if __name__ == "__main__":

	bf = ScalableBloomFilter(400, 0.001, 2, ScalableBloomFilter.SCALE_MODE_EXPONENTIAL)
	lines = open("/usr/share/dict/american-english").read().splitlines()
	for line in lines: bf.add(line)
	print "Max" in bf
	print "mice" in bf
	print "3" in bf
	print len(bf.filters)
	for f in bf.filters:
		print f.size, f.hash_count, f.element_count, f.calc_error_rate()

	#~ bf = BloomFilter(100000, 0.001)
	#~ bf2 = BloomFilter(100000, 0.001)
	#~ lines = open("/usr/share/dict/american-english").read().splitlines()
	#~ for line in lines:
		#~ if bf.element_count < 50000: bf.add(line)
		#~ else: bf2.add(line)
	#~ import datetime
	#~ start = datetime.datetime.now()
	#~ bf.lookup("google")
	#~ finish = datetime.datetime.now()
	#~ print (finish-start).microseconds
	#~ start = datetime.datetime.now()
	#~ bf.lookup("apple")
	#~ finish = datetime.datetime.now()
	#~ print (finish-start).microseconds
	#~ print "Max" in bf
	#~ print "mice" in bf
	#~ print "3" in bf
	#~ print bf.size, bf.hash_count, bf.element_count, bf.calc_error_rate()
	#~ print "Max" in bf2
	#~ print "mice" in bf2
	#~ print "3" in bf2
	#~ print bf2.size, bf2.hash_count, bf2.element_count, bf2.calc_error_rate()
	#~ bf3 = bf.union(bf2)
	#~ print "Max" in bf3
	#~ print "mice" in bf3
	#~ print "3" in bf3
	#~ print bf3.size, bf3.hash_count, bf3.element_count, bf3.calc_error_rate()

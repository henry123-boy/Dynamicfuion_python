// Copyright (C) 2005  Davis E. King (davis@dlib.net)
// License: Boost Software License   See https://github.com/davisking/dlib/blob/master/LICENSE.txt for the full license.
#pragma once
namespace test {

namespace eemk5 {
struct node {
	node* next;
	node* child_context;
	node* parent_context;

	unsigned short symbol;
	unsigned short count;
	unsigned short total;
	unsigned short escapes;
};
}


template<unsigned long alphabet_size, typename entropy_encoder, unsigned long total_nodes, unsigned long order>
class entropy_encoder_model_mantissa {
	/*!
		REQUIREMENTS ON total_nodes
			- 4096 < total_nodes
			- this is the total number of nodes that we will use in the tree

		REQUIREMENTS ON order
			- 0 <= order
			- this is the maximum depth-1 the tree will be allowed to go (note
			  that the root level is depth 0).

		GENERAL NOTES
			This implementation follows more or less the implementation
			strategy laid out by Alistair Moffat in his paper
			Implementing the PPM data compression scheme.  Published in IEEE
			Transactions on Communications, 38(11):1917-1921, 1990.

			The escape method used will be method D.

			This also uses Dmitry Shkarin's Information Inheritance scheme.
			(described in "PPM: one step to practicality" and "Improving the
			Efficiency of the PPM Algorithm")


		INITIAL VALUE
			- root == pointer to an array of total_nodes nodes
			- next_node == 1
			- cur == root
			- cur_order = 0
			- root->next == 0
			- root->parent_context == 0
			- root->child_context == 0
			- root->escapes == 0
			- root->total == 0
			- stack_size == 0
			- exc_used == false
			- for all i: exc[i] == 0

		CONVENTION
			- pop() == stack[stack_size-1].n and stack[stack_size-1].nc
			- exc_used == something_is_excluded()
			- is_excluded(symbol) == bit symbol&0x1F from exc[symbol>>5]
			- &get_entropy_encoder() == coder
			- root == pointer to an array of total_nodes nodes.
			  this is also the root of the tree.
			- if (next_node < total_nodes) then
				- next_node == the next node in root that has not yet been allocated

			- root->next == 0
			- root->parent_context == 0


			- for every node in the tree:
			  {
				- NOTATION:
					- The "context" of a node is the string of symbols seen
					  when you go from the root of the tree down (down though
					  child context pointers) to the node, including the symbol at
					  the node itself.  (note that the context of the root node
					  is "" or the empty string)
					- A set of nodes is in the same "context set" if all the node's
					  contexts are of length n and all the node's contexts share
					  the same prefix of length n-1.
					- The "child context set" of a node is a set of nodes with
					  contexts that are one symbol longer and prefixed by the node's
					  context.  For example, if a node has a context "abc" then the
					  nodes for contexts "abca", "abcb", "abcc", etc. are all in
					  the child context set of the node.
					- The "parent context" of a node is the context that is one
					  symbol shorter than the node's context and includes the
					  symbol in the node.  So the parent context of a node with
					  context "abcd" would be the context "bcd".


				- if (next != 0) then
					- next == pointer to the next node in the same context set
				- if (child_context != 0) then
					- child_context == pointer to the first node of the child
					  context set for this node.
					- escapes > 0
				- if (parent_context != 0) then
					- parent_context == pointer to the parent context of this node.
				- else
					- this node is the root node of the tree


				- if (this is not the root node) then
					- symbol == the symbol represented with this node
					- count == the number of times this symbol has been seen in its
					  parent context.
				- else
					- the root doesn't have a symbol.  i.e. the context for the
					  root node is "" or the empty string.

				- total == The sum of the counts of all the nodes
				  in the child context set + escapes.
				- escapes == the escape count for the context represented
				  by the node.
				- count > 0
			}


			- cur_order < order
			- cur_order == the depth of the node cur in the tree.
			  (note that the root node has depth 0)
			- cur == pointer to the node in the tree who's context matches
			  the most recent symbols we have seen.


	!*/

	typedef eemk5::node node;


public:

	typedef entropy_encoder entropy_encoder_type;

	explicit entropy_encoder_model_mantissa(entropy_encoder& coder);

	virtual ~entropy_encoder_model_mantissa();

	inline void clear();

	inline void encode(unsigned long symbol);

	entropy_encoder& get_entropy_encoder() { return coder; }

	static unsigned long get_alphabet_size() { return alphabet_size; }

	// deleted functions
	entropy_encoder_model_mantissa(
			entropy_encoder_model_mantissa<alphabet_size, entropy_encoder, total_nodes, order>&) = delete;        // copy constructor
	entropy_encoder_model_mantissa<alphabet_size, entropy_encoder, total_nodes, order>&
	operator=(entropy_encoder_model_mantissa<alphabet_size, entropy_encoder, total_nodes, order>&) = delete;    // assignment operator

private:

	inline eemk5::node* allocate_node();
	/*!
		requires
			- space_left() == true
		ensures
			- returns a pointer to a new node
	!*/

	inline bool space_left() const;
	/*!
		ensures
			- returns true if there is at least 1 free node left.
			- returns false otherwise
	!*/

	inline void exclude(unsigned short symbol);
	/*!
		ensures
			- #is_excluded(symbol) == true
			- #something_is_excluded() == true
	!*/

	inline bool something_is_excluded();
	/*!
		ensures
			- returns true if some symbol has been excluded.
			  returns false otherwise
	!*/

	inline bool is_excluded(unsigned short symbol);
	/*!
		ensures
			- if (symbol has been excluded) then
				- returns true
			- else
				- returns false
	!*/

	inline void clear_exclusions();
	/*!
		ensures
			- for all symbols #is_excluded(symbol) == false
			- #something_is_excluded() == true
	!*/

	inline void scale_counts(node* temp);
	/*!
		ensures
			- divides all the counts in the child context set of n by 2.
			- none of the nodes in the child context set will have a count of 0
	!*/

	inline void push(node* n, node* nc);
	/*!
		requires
			- stack_size < order
		ensures
			- #pop(a,b): a == n && b == nc
	!*/

	inline void pop(node*& n, node*& nc);
	/*!
		requires
			- stack_size > 0
		ensures
			- returns the two nodes at the top of the stack
	!*/

	struct nodes {
		node* n;
		node* nc;
	};

	unsigned long next_node;
	entropy_encoder& coder;
	node* root;
	node* cur;
	unsigned long cur_order;
	unsigned long exc[alphabet_size / 32 + 1]{};
	bool exc_used{};
	nodes stack[order + 1];
	unsigned long stack_size;


};

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
// member function definitions
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

template<
		unsigned long alphabet_size,
		typename entropy_encoder,
		unsigned long total_nodes,
		unsigned long order
>
entropy_encoder_model_mantissa<alphabet_size, entropy_encoder, total_nodes, order>::
entropy_encoder_model_mantissa(entropy_encoder& coder_) :
		next_node(1),
		coder(coder_),
		cur_order(0),
		stack_size(0) {
	static_assert(1 < alphabet_size && alphabet_size < 65535, "Failed assertion.");
	static_assert(4096 < total_nodes, "Failed assertion.");

	root = new node[total_nodes];
	cur = root;

	root->child_context = nullptr;
	root->escapes = 0;
	root->next = nullptr;
	root->parent_context = nullptr;
	root->total = 0;

	clear_exclusions();
}

// ----------------------------------------------------------------------------------------

template<
		unsigned long alphabet_size,
		typename entropy_encoder,
		unsigned long total_nodes,
		unsigned long order
>
entropy_encoder_model_mantissa<alphabet_size, entropy_encoder, total_nodes, order>::
~entropy_encoder_model_mantissa() {
	delete[] root;
}

// ----------------------------------------------------------------------------------------

template<
		unsigned long alphabet_size,
		typename entropy_encoder,
		unsigned long total_nodes,
		unsigned long order
>
void entropy_encoder_model_mantissa<alphabet_size, entropy_encoder, total_nodes, order>::
clear() {
	next_node = 1;
	root->child_context = nullptr;
	root->escapes = 0;
	root->total = 0;
	cur = root;
	cur_order = 0;
	stack_size = 0;

	clear_exclusions();
}

// ----------------------------------------------------------------------------------------

template<
		unsigned long alphabet_size,
		typename entropy_encoder,
		unsigned long total_nodes,
		unsigned long order
>
void entropy_encoder_model_mantissa<alphabet_size, entropy_encoder, total_nodes, order>::
encode(unsigned long sym) {
	auto symbol = static_cast<unsigned short>(sym);
	node* temp = cur;
	cur = nullptr;
	unsigned short low_count, high_count, total_count;
	node* new_node = nullptr;

	// local_order will track the level of temp in the tree
	unsigned long local_order = cur_order;


	unsigned short c; // c == t(a|sk)
	unsigned short t; // t == T(sk)


	if (something_is_excluded())
		clear_exclusions();

	while (true) {
		low_count = 0;
		high_count = 0;
		if (space_left()) {
			total_count = temp->total;

			if (total_count > 0) {
				// check if we need to scale the counts
				if (total_count > 10000) {
					scale_counts(temp);
					total_count = temp->total;
				}


				// find the symbol we are looking for and put a pointer to it
				// into found_symbol.  If it isn't found then found_symbol == 0.
				// also, low_count and high_count will be correctly set.
				node* n = temp->child_context;
				node* found_symbol = nullptr;
				node* last = nullptr;
				if (something_is_excluded()) {
					node* temp_last = nullptr;
					while (true) {
						if (is_excluded(n->symbol) == false) {
							exclude(n->symbol);
							if (found_symbol == nullptr) {
								high_count += n->count;
								if (n->symbol == symbol) {
									found_symbol = n;
									last = temp_last;
									low_count = high_count - n->count;
								}
							}
						} else {
							total_count -= n->count;
						}

						if (n->next == nullptr)
							break;
						temp_last = n;
						n = n->next;
					}
				} else {
					while (true) {
						high_count += n->count;
						exclude(n->symbol);

						if (n->symbol == symbol) {
							found_symbol = n;
							low_count = high_count - n->count;
							break;
						}

						if (n->next == nullptr)
							break;
						last = n;
						n = n->next;
					}
				}

				// if we found the symbol
				if (found_symbol) {
					n = found_symbol;
					if (new_node != nullptr) {
						new_node->parent_context = found_symbol;
					}


					coder.encode(low_count, high_count, total_count);
					c = n->count += 8;
					t = temp->total += 8;

					// move this node to the front
					if (last) {
						last->next = n->next;
						n->next = temp->child_context;
						temp->child_context = n;
					}


					if (cur == nullptr) {
						if (local_order >= order) {
							cur = n->parent_context;
							cur_order = local_order;
						} else {
							cur_order = local_order + 1;
							cur = n;
						}
					}

					break;

				} else { // if we hit the end of the context set without finding the symbol
					// finish excluding all the symbols
					while (n->next) {
						exclude(n->symbol);
						n = n->next;
					}

					if (new_node != nullptr) {
						new_node->parent_context = allocate_node();
						new_node = new_node->parent_context;
					} else {
						new_node = allocate_node();
					}

					n->next = new_node;

					// write an escape to a lower context
					coder.encode(high_count, total_count, total_count);
				}

			} else {  // if (total_count == 0)
				// this means that temp->child_context == 0 so we should make
				// a new node here.
				if (new_node != nullptr) {
					new_node->parent_context = allocate_node();
					new_node = new_node->parent_context;
				} else {
					new_node = allocate_node();
				}

				temp->child_context = new_node;
			}

			if (cur == nullptr && local_order < order) {
				cur = new_node;
				cur_order = local_order + 1;
			}

			// fill out the new node
			new_node->child_context = nullptr;
			new_node->escapes = 0;
			new_node->next = nullptr;
			new_node->total = 0;
			push(new_node, temp);

			if (temp != root) {
				temp = temp->parent_context;
				--local_order;
				continue;
			}

			t = 2056;
			c = 8;

			// since this is the root we are going to the order-(-1) context
			// so we can just take care of that here.
			new_node->parent_context = root;
			coder.encode(symbol, symbol + 1, alphabet_size);

			if (cur == nullptr) {
				cur = root;
				cur_order = 0;
			}
			break;
		} else {
			// there isn't enough space so we should throw away the tree
			clear();
			temp = cur;
			local_order = cur_order;
			cur = nullptr;
			new_node = nullptr;
		}
	} // while (true)


	// initialize the counts and symbol for any new nodes we have added
	// to the tree.
	node* n, * nc;
	while (stack_size > 0) {
		pop(n, nc);

		n->symbol = static_cast<unsigned short>(symbol);

		// if nc is not a deterministic context
		if (nc->total) {
			unsigned long temp2 = t - c + nc->total - nc->escapes - nc->escapes;
			unsigned long temp3 = nc->total;
			temp3 *= c;
			temp3 /= (temp2 | 1); // this or-ing by 1 is just to make sure that temp2 is never zero
			temp3 += 2;
			if (temp3 > 50000) temp3 = 50000;
			n->count = static_cast<unsigned short>(temp3);


			nc->escapes += 4;
			nc->total += static_cast<unsigned short>(temp3) + 4;
		} else {
			n->count = 3 + 5 * (c) / (t - c);

			nc->escapes = 4;
			nc->total = n->count + 4;
		}

		while (nc->total > 10000) {
			scale_counts(nc);
		}
	}
}

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
// private member function definitions
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

template<
		unsigned long alphabet_size,
		typename entropy_encoder,
		unsigned long total_nodes,
		unsigned long order
>
eemk5::node* entropy_encoder_model_mantissa<alphabet_size, entropy_encoder, total_nodes, order>::
allocate_node() {
	node* temp;
	temp = root + next_node;
	++next_node;
	return temp;
}

// ----------------------------------------------------------------------------------------

template<
		unsigned long alphabet_size,
		typename entropy_encoder,
		unsigned long total_nodes,
		unsigned long order
>
bool entropy_encoder_model_mantissa<alphabet_size, entropy_encoder, total_nodes, order>::
space_left() const {
	return (next_node < total_nodes);
}

// ----------------------------------------------------------------------------------------

template<
		unsigned long alphabet_size,
		typename entropy_encoder,
		unsigned long total_nodes,
		unsigned long order
>
void entropy_encoder_model_mantissa<alphabet_size, entropy_encoder, total_nodes, order>::
exclude(
		unsigned short symbol
) {
	exc_used = true;
	unsigned long temp = 1;
	temp <<= symbol & 0x1F;
	exc[symbol >> 5] |= temp;
}

// ----------------------------------------------------------------------------------------

template<
		unsigned long alphabet_size,
		typename entropy_encoder,
		unsigned long total_nodes,
		unsigned long order
>
bool entropy_encoder_model_mantissa<alphabet_size, entropy_encoder, total_nodes, order>::
is_excluded(
		unsigned short symbol
) {
	unsigned long temp = 1;
	temp <<= symbol & 0x1F;
	return ((exc[symbol >> 5] & temp) != 0);
}

// ----------------------------------------------------------------------------------------

template<
		unsigned long alphabet_size,
		typename entropy_encoder,
		unsigned long total_nodes,
		unsigned long order
>
void entropy_encoder_model_mantissa<alphabet_size, entropy_encoder, total_nodes, order>::
clear_exclusions(
) {
	exc_used = false;
	for (unsigned long i = 0; i < alphabet_size / 32 + 1; ++i) {
		exc[i] = 0;
	}
}

// ----------------------------------------------------------------------------------------

template<
		unsigned long alphabet_size,
		typename entropy_encoder,
		unsigned long total_nodes,
		unsigned long order
>
bool entropy_encoder_model_mantissa<alphabet_size, entropy_encoder, total_nodes, order>::
something_is_excluded(
) {
	return exc_used;
}

// ----------------------------------------------------------------------------------------

template<
		unsigned long alphabet_size,
		typename entropy_encoder,
		unsigned long total_nodes,
		unsigned long order
>
void entropy_encoder_model_mantissa<alphabet_size, entropy_encoder, total_nodes, order>::
push(
		node* n,
		node* nc
) {
	stack[stack_size].n = n;
	stack[stack_size].nc = nc;
	++stack_size;
}

// ----------------------------------------------------------------------------------------

template<
		unsigned long alphabet_size,
		typename entropy_encoder,
		unsigned long total_nodes,
		unsigned long order
>
void entropy_encoder_model_mantissa<alphabet_size, entropy_encoder, total_nodes, order>::
pop(
		node*& n,
		node*& nc
) {
	--stack_size;
	n = stack[stack_size].n;
	nc = stack[stack_size].nc;
}

// ----------------------------------------------------------------------------------------

template<
		unsigned long alphabet_size,
		typename entropy_encoder,
		unsigned long total_nodes,
		unsigned long order
>
void entropy_encoder_model_mantissa<alphabet_size, entropy_encoder, total_nodes, order>::
scale_counts(node* temp) {
	if (temp->escapes > 1)
		temp->escapes >>= 1;
	temp->total = temp->escapes;

	node* n = temp->child_context;
	while (n != nullptr) {
		if (n->count > 1)
			n->count >>= 1;

		temp->total += n->count;
		n = n->next;
	}
}

// ----------------------------------------------------------------------------------------
namespace eemk4 {
struct node {
	node* next;
	node* child_context;
	node* parent_context;

	unsigned short symbol;
	unsigned short count;
	unsigned short total;
	unsigned short escapes;
};
}


template<unsigned long alphabet_size, typename entropy_encoder, unsigned long total_nodes, unsigned long order>
class entropy_encoder_model_exponent {
	/*!
		REQUIREMENTS ON total_nodes
			- 4096 < total_nodes
			- this is the total number of nodes that we will use in the tree

		REQUIREMENTS ON order
			- 0 <= order
			- this is the maximum depth-1 the tree will be allowed to go (note
			  that the root level is depth 0).

		GENERAL NOTES
			This implementation follows more or less the implementation
			strategy laid out by Alistair Moffat in his paper
			Implementing the PPM data compression scheme.  Published in IEEE
			Transactions on Communications, 38(11):1917-1921, 1990.

			The escape method used will be method D.


		INITIAL VALUE
			- root == pointer to an array of total_nodes nodes
			- next_node == 1
			- cur == root
			- cur_order = 0
			- root->next == 0
			- root->parent_context == 0
			- root->child_context == 0
			- root->escapes == 0
			- root->total == 0

		CONVENTION
			- &get_entropy_encoder() == coder
			- root == pointer to an array of total_nodes nodes.
			  this is also the root of the tree.

			- if (next_node < total_nodes) then
				- next_node == the next node in root that has not yet been allocated

			- root->next == 0
			- root->parent_context == 0


			- for every node in the tree:
			  {
				- NOTATION:
					- The "context" of a node is the string of symbols seen
					  when you go from the root of the tree down (down though
					  child context pointers) to the node, including the symbol at
					  the node itself.  (note that the context of the root node
					  is "" or the empty string)
					- A set of nodes is in the same "context set" if all the node's
					  contexts are of length n and all the node's contexts share
					  the same prefix of length n-1.
					- The "child context set" of a node is a set of nodes with
					  contexts that are one symbol longer and prefixed by the node's
					  context.  For example, if a node has a context "abc" then the
					  nodes for contexts "abca", "abcb", "abcc", etc. are all in
					  the child context set of the node.
					- The "parent context" of a node is the context that is one
					  symbol shorter than the node's context and includes the
					  symbol in the node.  So the parent context of a node with
					  context "abcd" would be the context "bcd".


				- if (next != 0) then
					- next == pointer to the next node in the same context set
				- if (child_context != 0) then
					- child_context == pointer to the first node of the child
					  context set for this node.
				- if (parent_context != 0) then
					- parent_context == pointer to the parent context of this node.
				- else
					- this node is the root node of the tree


				- if (this is not the root node) then
					- symbol == the symbol represented with this node
					- count == the number of times this symbol has been seen in its
					  parent context.
				- else
					- the root doesn't have a symbol.  i.e. the context for the
					  root node is "" or the empty string.

				- total == The sum of the counts of all the nodes
				  in the child context set + escapes.
				- escapes == the escape count for the context represented
				  by the node.
			}


			- cur_order < order
			- cur_order == the depth of the node cur in the tree.
			  (note that the root node has depth 0)
			- cur == pointer to the node in the tree who's context matches
			  the most recent symbols we have seen.


	!*/

	typedef eemk4::node node;


public:

	typedef entropy_encoder entropy_encoder_type;

	explicit entropy_encoder_model_exponent(entropy_encoder& coder);

	virtual ~entropy_encoder_model_exponent();

	inline void clear();

	inline void encode(unsigned long symbol);

	entropy_encoder& get_entropy_encoder() { return coder; }

	static unsigned long get_alphabet_size() { return alphabet_size; }

	// deleted functions
	entropy_encoder_model_exponent(
			entropy_encoder_model_exponent<alphabet_size, entropy_encoder, total_nodes, order>&) = delete;        // copy constructor
	entropy_encoder_model_exponent<alphabet_size, entropy_encoder, total_nodes, order>&
	operator=(entropy_encoder_model_exponent<alphabet_size, entropy_encoder, total_nodes, order>&) = delete;    // assignment operator

private:

	inline eemk4::node* allocate_node();
	/*!
		requires
			- space_left() == true
		ensures
			- returns a pointer to a new node
	!*/

	inline void destroy_tree();
	/*!
		ensures
			- deallocates all nodes except the root
			- #root->child_context == 0
			- #root->escapes == 0
			- #root->total == 0
			- #cur == root
			- #cur_order == 0
	!*/


	inline bool space_left() const;
	/*!
		ensures
			- returns true if there is at least 1 free node left.
			- returns false otherwise
	!*/


	inline void scale_counts(node* temp);
	/*!
		ensures
			- divides all the counts in the child context set of n by 2.
			- none of the nodes in the child context set will have a count of 0
	!*/


	unsigned long next_node;
	entropy_encoder& coder;
	node* root;
	node* cur;
	unsigned long cur_order;


};

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
// member function definitions
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

template<
		unsigned long alphabet_size,
		typename entropy_encoder,
		unsigned long total_nodes,
		unsigned long order
>
entropy_encoder_model_exponent<alphabet_size, entropy_encoder, total_nodes, order>::
entropy_encoder_model_exponent(entropy_encoder& coder_) :
		next_node(1),
		coder(coder_),
		cur_order(0) {
	static_assert(1 < alphabet_size && alphabet_size < 65535, "Failed assertion.");
	static_assert(4096 < total_nodes, "Failed assertion.");

	root = new node[total_nodes];
	cur = root;

	root->child_context = nullptr;
	root->escapes = 0;
	root->next = nullptr;
	root->parent_context = nullptr;
	root->total = 0;
}

// ----------------------------------------------------------------------------------------

template<
		unsigned long alphabet_size,
		typename entropy_encoder,
		unsigned long total_nodes,
		unsigned long order
>
entropy_encoder_model_exponent<alphabet_size, entropy_encoder, total_nodes, order>::
~entropy_encoder_model_exponent() { delete[] root; }

// ----------------------------------------------------------------------------------------

template<
		unsigned long alphabet_size,
		typename entropy_encoder,
		unsigned long total_nodes,
		unsigned long order
>
void entropy_encoder_model_exponent<alphabet_size, entropy_encoder, total_nodes, order>::
clear() { destroy_tree(); }

// ----------------------------------------------------------------------------------------

template<
		unsigned long alphabet_size,
		typename entropy_encoder,
		unsigned long total_nodes,
		unsigned long order
>
void entropy_encoder_model_exponent<alphabet_size, entropy_encoder, total_nodes, order>::
encode(unsigned long sym) {
	auto symbol = static_cast<unsigned short>(sym);
	node* temp = cur;
	cur = nullptr;
	unsigned short low_count, high_count, total_count;
	node* new_node = nullptr;

	// local_order will track the level of temp in the tree
	unsigned long local_order = cur_order;

	while (true) {
		high_count = 0;
		if (space_left()) {
			total_count = temp->total;

			if (total_count > 0) {
				// check if we need to scale the counts
				if (total_count > 10000) {
					scale_counts(temp);
					total_count = temp->total;
				}

				// find either the symbol we are looking for or the
				// end of the context set
				node* n = temp->child_context;
				node* last = nullptr;
				while (true) {
					high_count += n->count;

					if (n->symbol == symbol || n->next == nullptr)
						break;
					last = n;
					n = n->next;
				}

				low_count = high_count - n->count;

				// if we found the symbol
				if (n->symbol == symbol) {
					if (new_node != nullptr) {
						new_node->parent_context = n;
					}

					coder.encode(low_count, high_count, total_count);
					n->count += 8;
					temp->total += 8;


					// move this node to the front
					if (last) {
						last->next = n->next;
						n->next = temp->child_context;
						temp->child_context = n;
					}


					if (cur == nullptr) {
						if (local_order < order) {
							cur_order = local_order + 1;
							cur = n;
						} else {
							cur = n->parent_context;
							cur_order = local_order;
						}
					}

					break;

				}
					// if we hit the end of the context set without finding the symbol
				else {
					if (new_node != nullptr) {
						new_node->parent_context = allocate_node();
						new_node = new_node->parent_context;
					} else {
						new_node = allocate_node();
					}

					n->next = new_node;

					// write an escape to a lower context
					coder.encode(high_count, total_count, total_count);
				}

			} else // if (total_count == 0)
			{
				// this means that temp->child_context == 0 so we should make
				// a new node here.
				if (new_node != nullptr) {
					new_node->parent_context = allocate_node();
					new_node = new_node->parent_context;
				} else {
					new_node = allocate_node();
				}

				temp->child_context = new_node;
			}

			if (cur == nullptr && local_order < order) {
				cur = new_node;
				cur_order = local_order + 1;
			}

			// fill out the new node
			new_node->child_context = nullptr;
			new_node->count = 4;
			new_node->escapes = 0;
			new_node->next = nullptr;
			new_node->symbol = static_cast<unsigned short>(symbol);
			new_node->total = 0;
			temp->escapes += 4;
			temp->total += 8;


			if (temp != root) {
				temp = temp->parent_context;
				--local_order;
				continue;
			}

			// since this is the root we are going to the order-(-1) context
			// so we can just take care of that here.
			new_node->parent_context = root;
			coder.encode(symbol, symbol + 1, alphabet_size);

			if (cur == nullptr) {
				cur = root;
				cur_order = 0;
			}
			break;
		} else {
			// there isn't enough space so we should rebuild the tree
			destroy_tree();
			temp = cur;
			local_order = cur_order;
			cur = nullptr;
			new_node = nullptr;
		}
	} // while (true)

}

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
// private member function definitions
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

template<
		unsigned long alphabet_size,
		typename entropy_encoder,
		unsigned long total_nodes,
		unsigned long order
>
eemk4::node* entropy_encoder_model_exponent<alphabet_size, entropy_encoder, total_nodes, order>::
allocate_node() {
	node* temp;
	temp = root + next_node;
	++next_node;
	return temp;
}

// ----------------------------------------------------------------------------------------

template<
		unsigned long alphabet_size,
		typename entropy_encoder,
		unsigned long total_nodes,
		unsigned long order
>
void entropy_encoder_model_exponent<alphabet_size, entropy_encoder, total_nodes, order>::
destroy_tree() {
	next_node = 1;
	root->child_context = nullptr;
	root->escapes = 0;
	root->total = 0;
	cur = root;
	cur_order = 0;
}

// ----------------------------------------------------------------------------------------

template<
		unsigned long alphabet_size,
		typename entropy_encoder,
		unsigned long total_nodes,
		unsigned long order
>
bool entropy_encoder_model_exponent<alphabet_size, entropy_encoder, total_nodes, order>::
space_left() const {
	return (next_node < total_nodes);
}

// ----------------------------------------------------------------------------------------

template<
		unsigned long alphabet_size,
		typename entropy_encoder,
		unsigned long total_nodes,
		unsigned long order
>
void entropy_encoder_model_exponent<alphabet_size, entropy_encoder, total_nodes, order>::
scale_counts(
		node* temp
) {
	if (temp->escapes > 1)
		temp->escapes >>= 1;
	temp->total = temp->escapes;

	node* n = temp->child_context;
	while (n != nullptr) {
		if (n->count > 1)
			n->count >>= 1;

		temp->total += n->count;
		n = n->next;
	}
}

// ----------------------------------------------------------------------------------------

} // namespace test
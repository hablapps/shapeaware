# Shape-dependent computations in Scala ... and Agda!

In this post we will solve a little programming problem, mainly with the excuse of talking about dependent types. As usual, Scala will be our programming language of choice. However, this time we will also use [Agda](https://agda.readthedocs.io/en/latest/index.html), a programming language which boasts full-fledged support for dependent types. The ultimate goal of this post is comparing both implementations and sharing our experiences with dependently typed programming.

Let’s start with ...

## Our little problem

Let’s consider the following type of (non-empty) binary trees, implemented in Scala as a common algebraic data type:

```scala
sealed abstract class Tree[A]
case class Leaf[A](a : A) extends Tree[A]
case class Node[A](left: Tree[A], root: A, right: Tree[A]) extends Tree[A]
```

We want to implement two functions that allow us to get and update the leafs of a given tree. As a first attempt (there will be several attempts more before we reach the solution, be patient!), we may come about with the following signatures:

```scala
class Leafs[A]{
  def get(tree: Tree[A]): List[A] = ???
  def update(tree: Tree[A]): List[A] => Tree[A] = ???
}
```

The `get` function bears no problem: there may be one or several leafs in the input tree, and the resulting list can cope with that. The `update` function, however, while essentially being what we want, poses some problems. This method returns a function which updates the leafs of the tree given a list of new values for those nodes. Ideally, we would expect to receive a list with exactly as many values as leafs are there in the tree. But given this signature, this may not happen at all: we may receive less values or more. In the former case, we are forced to make a choice: either to return the original tree or throwing an exception (abandoning purity). In the latter, it would be fair to return the exceeding values, besides the updated tree. In sum, the following signature seems to be more compliant with the problem at hand:

```scala
class Leafs[A]{
  def get(tree: Tree[A]): List[A] = ???
  def update(tree: Tree[A]): List[A] => Option[(List[A], Tree[A])] = ???
}
```

Essentially, the `update` method now returns a stateful computation, i.e. a value of the famous [`StateT`](https://typelevel.org/cats/datatypes/state.html) monad. This computation is run by giving an initial list of values, and will finish with a value `None` (meaning that it couldn’t complete the computation) or `Some(l, t)`, i.e. the updated tree `t` and the list of exceeding values `l` (possibly, empty). We won’t show the implementation of these methods, but you can find it in the [repository](https://github.com/hablapps/shapeaware/blob/master/src/test/scala/code.scala#L26) of this post.

Ok, this is nice, but we are stubborn and keep insisting on finding a way to prevent the user to pass a wrong number of values to the `update` method. I mean, we want to program the signature in such a way that the compiler throws an error if the programmer tries to call our function with less or more values than needed. Is it that possible?

<h2>Solving the problem with dependent types</h2>

A possible signature that solves our problem is the following one:

```scala
def update(tree: Tree[A]): Vec[A, n_leafs(tree)] => Tree[A]
```

where `n_leafs: Tree[A] => Integer` is a function that returns the number of leafs of the specified tree, and the `Vec` type represents lists of a fixed size. This signature gives the Scala compiler the required information to grant execution of the following call:

```scala
scala> update(Node(Leaf(1), 2, Leaf(3)))(Vec(3, 1))
res11: Tree[Int] = Node(Leaf(3), 2, Leaf(1))
```

and block the following one instead, with a nice compiler error:

```scala
scala> update(Node(Leaf(1), 2, Leaf(3)))(Vec(3))
:18: error: type mismatch;
 found   : Vec[Int, 1]
 required: Vec[Int, 2]
       update(Node(Leaf(1), 2, Leaf(3)))(Vec(3))
```

... wouldn’t this be beautiful?

Alas, the above signature is not legal Scala 2.12. The problem is in the `Vec[? , ? : Nat]` type constructor. As we said, it holds two parameters. There is no problem with the first one: type constructors in Scala do indeed receive types as arguments. Another way of saying this is that types in Scala can be parameterised with respect to types. And yet another way is saying that types in Scala can be made *dependent* on types. But the second parameter of the `Vec` constructor is not a type, it’s a *value*! And we can’t parameterise types in Scala with respect to values, only to types.

A type whose definition refers to values is called a *dependent type*. Indeed, the type `List[A]` in Scala also *depends* on something, to wit the type `A`. So, in a sense, we may rightfully call it a dependent type as well. However, the “dependent” qualifier is conventionally reserved for types that are parameterised with respect to values.

Can’t we solve our problem in Scala, then? Yes, we will see that we can indeed solve this problem in Scala, albeit in a differnt way. But before delving into the Scala solution, let’s see how we can solve this problem in a language with full-fledged dependent types, in line with the solution sketched at the beginning of this section.

<h2>The solution in Agda</h2>

First, we must define the tree data type:

```haskell
module Trees where
  data Tree (A : Set) : Set where
    leaf : A -> Tree A
    node : Tree A -> A -> Tree A -> Tree A
```

This a common algebraic data type definition, with constructors `leaf` and `node`. The definition is parameterised with respect to `A`, which is declared to be a regular type, i.e. `Set`. The resulting type `Tree A` is also a regular type (i.e. not a type constructor, which would be declared as `Set -> Set`). Next, we have to define the following function:

```haskell
  open import Data.Nat

  n_leafs : {A : Set} -> Tree A -> ℕ
  n_leafs (leaf _) = 1
  n_leafs (node l _ r) = n_leafs l + n_leafs r
```

The `n_leafs` function returns the number of leafs held by a given tree (as a natural number ℕ declared in the `Data.Nat` module). The implementation is based on pattern matching, using the same underscore symbol that we use in Scala whenever we are not interested in some value.


Let’s implement now the promised `get` and `update` functions, which will be part of a module named `Leafs`:

```haskell
module Leafs where

  open import Data.Vec
  open Trees

  get : {A : Set} -> (s : Tree A) -> Vec A (n_leafs s) = ?
  update : {A : Set} -> (s : Tree A) -> Vec A (n_leafs s) -> Tree A = ?
```

As you can see, we can now use the `n_leafs s` value in a type definition! Indeed, the `Vec (A : Set) (n : ℕ)` type is a truly dependent type. It represents lists of values of a fixed size `n`. Moreover, the size does not need to be a constant such as 1, 2, 3, etc. It can be the result of a function, as this example shows. The implications of this are huge, as we will soon realise.

Let’s expand the definition of the `get` function:

```haskell
  get : {A : Set} -> (s : Tree A) -> Vec A (n_leafs s)
  get (leaf x) = x ∷ []
  get (node l _ r) = get l ++ get r
```

If the tree is a leaf, we just return its value in a vector of length one. Otherwise, we collect recursively the leafs of the left and right subtrees and return their concatenation. What would happen if we implemented the first clause in the pattern matching as `get (leaf x) = []` (i.e. if we attempted to return the empty vector for a leaf tree)? The compiler would complain with the following error:

```haskell
0 != 1 of type .Agda.Builtin.Nat.Nat
when checking that the expression [] has type
Vec .A (n_leafs (leaf x))
```

This error says that 0, i.e. the length of the empty vector `[]`, does not equal 1, i.e. the number of leafs of the input tree `leaf x`. All this while attempting to check that the proposed output `[]`, whose type is `Vec A 0`, has the required type `Vec .A (n_leafs (leaf x))`, i.e. `Vec A 1`. Similarly, in the second clause, the compiler will care itself to check that `n_leafs l + n_leafs r`, which is the resulting length of the vector concatenation `get l :: get r`, equals the value `n_leafs (node l _ r)`, which according to the definition of the `n_leafs` function is indeed the case. In sum, we can’t cheat the compiler and return a vector with a number of values different to the number of leafs in the input tree. This property is hardwired in the signature of the function, thanks to the expressiveness of the Agda type system. And to be able to guarantee that, Agda needs to be able to perform computations on values at compile time.

The implementation of the `update` function is similarly beautiful:

```haskell
update : {A : Set} -> (s : Tree A) -> Vec A (n_leafs s) -> Tree A
  update (leaf _) (x ∷ []) = leaf x
  update (node l x r) v = node updatedL x updatedR
    where
      updatedL = update l (take (n_leafs l) v)
      updatedR = update r (drop (n_leafs l) v)
```

Note that in the first clause of the pattern matching, we were able to deconstruct the input vector into the shape `x ∷ []`, without the compiler complaining about missing clauses for the `leaf` constructor. This is because Agda knows (by evaluating the `n_leafs` function) that any possible leaf tree has a number of leafs equals to one. In the second clause, the input vector has type `v : Vec A (n_leafs (node l x r))`, which Agda knows to be `v : Vec A (n_leafs l + n_leafs r)` by partially evaluating the `n_leafs` function. This is what makes the subsequent calls to update the left and right subtrees typesafe. Indeed, to update the left subtree `l` we need a vector with a number of elements equal to its number of leafs `n_leafs l`. This vector has to be a subvector of the input vector `v`, which Agda knows to have length `n_leafs l + n_leafs r` as we mentioned before. So, the expression `take (n_leafs l) v` will compile without problems. Similarly, Agda knows that the length of the `drop (n_leafs l) v` vector will be `n_leafs r` (by checking the definition of the concatenation function `++`), which is precisely what the `update r` function needs.

Let’s exercise these definitions in the following module:

```haskell
  module TestLeafs where
    open import Data.Nat
    open import Data.Vec
    open Trees
    open LeafsAdHoc

    t1 : Tree ℕ
    t1 = node (node (leaf 1) 2 (leaf 3)) 4 (leaf 5)

    l1 : Vec ℕ 3
    l1 = Leafs.get t1

    t2 : Tree ℕ
    t2 = Leafs.update t1 (5 ∷ 3 ∷ 1 ∷ [])

    // CHECK

    open import Relation.Binary.PropositionalEquality

    eq1 : l1 ≡ (1 ∷ 3 ∷ 5 ∷ [])
    eq1 = refl

    eq2 : t2 ≡ (node (node (leaf 5) 2 (leaf 3)) 4 (leaf 1))
    eq2 = refl

    -- WON'T COMPILE

    {- Error: 3 != 4 of type ℕ
       when checking that the expression get t1 has type Vec ℕ 4

    l2 : Vec ℕ 4
    l2 = Leafs.get t1
    -}

    {- Error: 0 != 2 of type ℕ
       when checking that the expression [] has type Vec ℕ 2

    t3 : Tree ℕ
    t3 = Leafs.update t1 (5 ∷ [])
    -}
```

The `l1` variable represents the leafs of the sample tree `t1`, namely values 1, 3 and 5. Accordingly, the type of the variable is `Vec ℕ 3`. The variable `t2` is the result of updating the tree with a new collection of leafs. In both cases, we make reference to the functions `get` and `update` declared in the module `Leafs`.

The next lines *prove* that the values of these variables are the expected ones, making use of the equality type constructor `_≡_` and its `refl` constructor (note that `_≡_` is parameterised with respect two values, so it's a dependent type). The proof is plain `refl`exivity, i.e. `x ≡ x`, since `l1` and `t2` actually evaluate to the same values.

Note that the fact that this code compiles is enough to show that the tests pass. We don’t need to run anything! On the other hand, Agda allows us to test that our functions work as expected by implementing much more complex proofs for more expressive properties. We will leave that for another post.

Let’s come back to Scala.

<h2>The solution in Scala</h2>

We can’t make computations on values in Scala at compile time, but we can do it on types! And this suffices to solve our problem, albeit in a different form to Agda. We will reconcile both approaches in the next section.

Type-level computation in Scala proceeds through the implicits mechanism. But before we can exploit implicits, we first need to re-implement our `Tree` data type so that we don’t loose the *shapes* of trees:

```scala
sealed abstract class Tree[A]
case class Leaf[A](value: A) extends Tree[A]
case class Node[L <: Tree[A], A, R <: Tree[A]](
  left: L, root: A, right: R) extends Tree[A]
```

This new implementation differs with the previous one in the types of the recursive arguments of the `Node` constructor. Now, they are generic parameters `L` and `R`, declared to be subtypes of `Tree[A]`, i.e. either leafs or nodes. Essentially, this allows us to preserve the exact type of the tree; what we will call its *shape*. In essence, this is the same trick commonly used to implement heterogeneous lists in Scala (see, e.g. their [implementation](https://github.com/milessabin/shapeless/blob/master/core/src/main/scala/shapeless/hlists.scala#L30) in the shapeless framework).

For instance, let’s compare both implementations in the REPL, with the old implementation of the `Tree` data type located in the `P` module, and the new one in the current scope:

```scala
scala> val p_tree = P.Node(P.Node(P.Leaf(1), 2, P.Leaf(3)), 4, P.Leaf(5))
p_tree: P.Node[Int] = ...

scala> val tree = Node(Node(Leaf(1), 2, Leaf(3)), 4, Leaf(5))
tree: Node[Node[Leaf[Int], Int, Leaf[Int]], Int, Leaf[Int]] = ...
```

As we can see, the type of `p_tree` is simply `Node[Int]`, whereas the type of `tree` is much more informative: we don’t only know that it is a node tree; we know that it holds exactly five elements, three of which are leafs. Its shape has not been lost.

We can apply the same trick to the `List` type, in order to preserve information about the shape of list instances (essentially, how many values it stores). This is the resulting definition:

```scala
sealed abstract class List[A]
case class Nil[A]() extends List[A]
case class Cons[A, T <: List[A]](head: A, tail: T) extends List[A]
```

Let's see now how can we exploit these shape-aware, algebraic data types, to support shape-dependent, type-level computations … and finally solve our little problem. Recall the original signatures for the `get/update` functions, which built upon the common, non-shape aware definitions of the `Tree` and `List` data types:

```scala
class Leafs[A]{
  def get(tree: Tree[A]): List[A] = ???
  def update(tree: Tree[A]): List[A] => Tree[A] = ???
}
```

Now we can explain their limitations in a more precise way. For instance, let’s consider the resulting function of `update`. The input of this function is declared to be any `List[A]`, not lists of a particular *shape*. That’s relevant to our problem because we want the compiler to be able to block invocations for trees of an undesired shape, i.e. length. But how can we represent the shape of an algebraic data type in the Scala type system? The answer is *subtyping*, i.e. we can declare the result of that function to be some `L <: List[A]`, instead of a plain `List[A]`. There is a one-to-one correspondence between the subtypes of the algebraic data type `List[A]` and its possible shapes.

Similarly, the input trees of `get` and `update` are declared to be any `Tree[A]`, instead of trees of a particular shape `T <: Tree[A]`. This is bad, because in that way we won’t be able to determine which is the exact list shape that must be returned for a given tree. Ok, but how can we determine the shape of list corresponding to a given shape of tree? The answer is using *type-level functions* which operates on input/output types that represent shapes.

These shape-dependent functions are declared as traits and defined through the implicits mechanism. For instance, the declaration of the type-level function between trees and lists is as follows:

```scala
trait LeafsShape[In <: Tree[A]]{
  type Out <: List[A]

  def get(t: In): Out
  def update(t: In): Out => In
}
```

The `LeafsShape` trait is parameterised with respect to any *shape* of tree. Its instance for a particular shape will give us the list shape that we can use to store the current leafs of the tree, or the new values required for those leafs. Moreover, for that particular shape of tree we also obtain its corresponding get and update implementations.

Concerning the implementation of the shape-dependent function `LeafsShape`, i.e. how do we compute the shape of list corresponding to a given shape of tree, we proceed through implicits defined in its companion object. The following signatures (not for the faint of heart …) suffice:

```scala
object LeafsShape{
  type Output[T <: Tree[A], _Out] = LeafsShape[T]{ type Out = _Out }

  implicit def leafCase: Output[Leaf[A], Cons[A, Nil[A]]] = ???

  implicit def nodeCase[
    L <: Tree[A],
    R <: Tree[A],
    LOut <: List[A],
    ROut <: List[A]](implicit
    ShapeL: Output[L, LOut],
    ShapeR: Output[R, ROut],
    Conc: Concatenate[A, LOut, ROut]
  ): Output[Node[L, A, R], Conc.Out] = ???
```

We omit the implementations of the `get` and `update` functions to focus on the list shape computation, which is shown through the type alias `Output`. The first case is easy: the shape of list which we need to hold the leafs of a tree of type `Leaf[A]` is the one that allows us to store a single element of type `A`, i.e. `Cons[A, Nil[A]]`. For arbitrary node trees, the situation is in appearance more complicated, though conceptually simple. Given a tree of shape `Node[L, A, R]`, we first need to know the list shapes for the left and right subtrees `L` and `R`. The implicit arguments `ShapeL` and `ShapeR` provide us with the `LOut` and `ROut` shapes. The resulting list shape will be precisely their concatenation, which we achieve through an auxiliary type-level function `Concatenate` (not shown for brevity, but implemented in a similar way). The shape concatenation will be accessible through the `Out` type member variable of that function. The `Conc.Out` type is an example of path-dependent type, a truly dependent type since it depends on the value `Conc` obtained through the implicits mechanism.

We are about to finish. The only thing that is needed is some way to call the `get` and `update` member functions of the `LeafsShape` type-level function, for a given tree value. We achieve that with two auxiliary definitions, located in a definitive `Leafs` module (where the type-level function and its companion object are also implemented):

```scala
class Leafs[A]{
  def get[In <: Tree[A]](t : In)(implicit S: LeafsShape[In]): S.Out = S.get(t)
  def update[In <: Tree[A]](t : In)(implicit S: LeafsShape[In]): S.Out => In = S.update(t)

  trait LeafsShape[In <: Tree[A]]{ ... }
  object LeafsShape{ ... }
}
```

The auxiliary functions `get` and `update` are the typesafe counterparts of the original signatures. The first difference that we may emphasise is that the type of input trees is not a plain, uninformative `Tree[A]`, but a particular shape of tree `In`. The compiler can then use this shape as input to the type-level function `LeafsShape`, to compute the shape of the resulting list `S.Out`. The output of these functions is thus declared as a path-dependent type. Last, note that the implementation of these functions is wholly delegated to the corresponding implementations of the inferred type-level function.

Let’s see how this works in the following REPL session:

```scala
scala> val tree = Node(Node(Leaf(1), 2, Leaf(3)), 4, Leaf(5))
tree: Node[Node[Leaf[Int], Int, Leaf[Int]], Int, Leaf[Int]] = ...

scala> get(tree)
res2: Cons[Int, Cons[Int, Cons[Int, Nil[Int]]]] = Cons(1,Cons(3,Cons(5,Nil())))

scala> update(tree).apply(Cons(5, Cons(3, Cons(1, Nil[Int]()))))
res1: Node[Node[Leaf[Int], Int, Leaf[Int]], Int, Leaf[Int]] =
  Node(Node(Leaf(5), 2, Leaf(3)), 4, Leaf(1))

scala> update(tree).apply(Cons(5, Nil[Int]()))
:22: error: type mismatch;
 found   : Nil[Int]
 required: Cons[Int, Cons[Int, Nil[Int]]]
       update(tree).apply(Cons(5, Nil[Int]())
                                          ^
```

As expected, when we pass lists of the right shape, everything works. On the contrary, as shown in the last example, if we pass a list of the wrong size, the compiler will complain. In particular, the error message tells us that it found a list of type `Nil[Int]` where it expected a list of size two. This is because `update(tree)` returns a list of shape three, and we only pass a list of size one. This is exactly the same behaviour that we got with the Agda implementation.

<h2>Reconciling Scala and Agda</h2>

The Scala and Agda implementations seem very different. In Scala, we exploit the expressiveness of its type system to preserve the shape of algebraic data type values, and perform type-level, shape-dependent computations at compile time. In Agda, we exploit its capability to declare full-fledged dependent types, and perform value-level computations at compile time.

Nonetheless, let’s recall the signatures of both implementations and try to reconcile their differences:

```haskell
-- AGDA VERSION

module Leafs where
  open import Data.Vec
  open Trees

  get : {A : Set} -> (s : Tree A) -> Vec A (n_leafs s) = ?
  update : {A : Set} -> (s : Tree A) -> Vec A (n_leafs s) -> Tree A = ?
```

```scala
// SCALA VERSION

class Leafs[A]{
  def get[In <: Tree[A]](t : In)(implicit S: LeafsShape[In]): S.Out = S.get(t)
  def update[In <: Tree[A]](t : In)(implicit S: LeafsShape[In]): S.Out => In = S.update(t)

  trait LeafsShape[In <: Tree[A]]{
    type Out <: List[A]

    def get(t: In): Out
    def update(t: In): Out => In
  }

  object LeafsShape{ ... }
}
```

In a sense, the Scala signature is simpler: there is no need to use a different type `Vec (A : Set) n : Nat`. The very same algebraic data type `List[A]` (albeit implemented in a shape-aware fashion), and subtyping suffice for representing shapes. In Agda, the new vector type is introduced precisely to represent the shapes of lists, which are in one to one correspondence with the natural numbers.

The `#length` function is then used to compute the required shape for a given tree. In Scala, there is no particular need for that, since the shape is computed along the implementation of the `get` and `update` functions in the type-level function `LeafsShape`.

The downside of the Scala implementation is, evidently, its verbosity and the amount of techniques and tricks involved: path-dependent types, traits, subtyping, implicits, auxiliary functions, … This is a <a href="https://github.com/lampepfl/dotty/pull/3844">recognised problem</a> which is being tackled for the future Scala 3.0 version.

<h2>Conclusion</h2>

We may have mimicked the Agda implementation style in Scala. In the `shapeless` framework, for instance, we have available the `Sized` and `Nat` types to represent lists of a fixed size (see the implementation [here](https://github.com/hablapps/shapeaware/blob/master/src/test/scala/code.scala#L207)), and we may even use <a href="https://docs.scala-lang.org/sips/42.type.html">literal types</a> to overcome the limitation of using values in type declarations. Alternatively, we proposed an implementation fully based on shape-aware algebraic data types. This version is in our opinion more idiomatic to solve our particular problem in Scala. It also allows us to grasp the idiosyncrasy of Scala with respect to competing approaches like the one proposed in Agda. In this regard, we found the notion of <a href="http://www.cs.nott.ac.uk/~psztxa/publ/fossacs03.pdf">*shape*</a> to be extremely useful.


In next posts we will likely go on exploring Agda in one of its most characteristic applications: certified programming. For instance, we may generalise the example shown in this post and talk about *traversals* (a kind of optic, like lenses) and its laws. One of these laws, applied to our example, tells us that if you update the leafs of the tree with its current leaf values, you will obtain the same tree. Using Agda, we can state that law and *prove* that our implementation satisfies it. No need to enumerate test cases, or empirically test the given property (e.g., as in Scalacheck). Till the next post!

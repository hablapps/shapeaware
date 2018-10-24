module code where

  module Trees where

    data Tree (A : Set) : Set where
      leaf : A -> Tree A
      node : Tree A -> A -> Tree A -> Tree A

    open import Data.Nat

    n_leafs : {A : Set} -> Tree A -> ℕ
    n_leafs (leaf _) = 1
    n_leafs (node l _ r) = n_leafs l + n_leafs r

  module Leafs where

    open import Data.Vec
    open Trees

    get : {A : Set} -> (s : Tree A) -> Vec A (n_leafs s)
    get (leaf x) = x ∷ []
    get (node l _ r) = get l ++ get r

    update : {A : Set} -> (s : Tree A) -> Vec A (n_leafs s) -> Tree A
    update (leaf _) (x ∷ []) = leaf x
    update (node l x r) v = node updatedL x updatedR
      where
        updatedL = update l (take (n_leafs l) v)
        updatedR = update r (drop (n_leafs l) v)

  module TestLeafs where

    open import Relation.Binary.PropositionalEquality
    open import Data.Nat
    open import Data.Vec

    open Trees
    open Leafs

    t1 : Tree ℕ
    t1 = node (node (leaf 1) 2 (leaf 3)) 4 (leaf 5)

    l1 : Vec ℕ 3
    l1 = Leafs.get t1

    t2 : Tree ℕ
    t2 = Leafs.update t1 (5 ∷ 3 ∷ 1 ∷ [])

    -- CHECK
    
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


package org.hablapps.shapeaware

import org.scalatest._

/*
THE PROBLEM
*/

object TheProblem extends TheProblem
class TheProblem extends FunSpec with Matchers{

  sealed abstract class Tree[A]
  case class Leaf[A](a : A) extends Tree[A]
  case class Node[A](left: Tree[A], root: A, right: Tree[A]) extends Tree[A]

  import cats.data.StateT, cats.instances.option._, cats.syntax.apply._

  class FirstAttemptLeafs[A]{

    def get(tree: Tree[A]): List[A] = ???

    def update(tree: Tree[A]): List[A] => Tree[A] = ???
  }

  class Leafs[A]{

    def get(tree: Tree[A]): List[A] = tree match {
      case Leaf(root) => List(root)
      case Node(left, _, right) => get(left) ++ get(right)
    }

    def update(tree: Tree[A]): StateT[Option, List[A], Tree[A]] = tree match {
      case Leaf(_) => for {
        (head :: tail) <- StateT.get[Option, List[A]]
        _ <- StateT.set(tail)
      } yield Leaf(head)

      case Node(left, root, right) =>
        update(left).map2(update(right))(Node(_, root, _))
    }
  }

  val leafsInt = new Leafs[Int]
  import leafsInt._

  describe("test1"){
    val t: Tree[Int] = Node(Leaf(1), 2, Leaf(3))

    val c: StateT[Option, List[Int], Tree[Int]] = update(t)

    it("Ok"){
      c.runA(List(3, 4)) shouldBe Some(Node(Leaf(3), 2, Leaf(4)))
    }

    it("No Ok"){
      c.runA(List(3)) shouldBe None
    }

    it("Exceeds"){
      c.run(List(3,4,5)) shouldBe Some((List(5), Node(Leaf(3), 2, Leaf(4))))
    }
  }
}

object TheSolutionInScala extends TheSolutionInScala
class TheSolutionInScala extends FunSpec with Matchers{ self =>

  sealed abstract class Tree[A]
  case class Leaf[A](value: A) extends Tree[A]
  case class Node[L <: Tree[A], A, R <: Tree[A]](left: L, root: A, right: R) extends Tree[A]

  sealed abstract class List[A]
  case class Nil[A]() extends List[A]
  case class Cons[A, T <: List[A]](head: A, tail: T) extends List[A]

  trait Concat[A, L1 <: List[A], L2 <: List[A]]{
    type Out <: List[A]

    def apply(l1 : L1, l2 : L2): Out
    // Can't use proper unapply due to SI-9247
    def unapply(l: Out): (L1, L2)
  }

  object Concat{

    type Output[A, L1 <: List[A], L2 <: List[A], _Out <: List[A]] =
      Concat[A, L1, L2]{ type Out = _Out }

    implicit def concatNil[A, L2 <: List[A]]: Concat.Output[A, Nil[A], L2, L2] =
      new Concat[A, Nil[A], L2]{
        type Out = L2

        def apply(l1 : Nil[A], l2 : L2) = l2
        def unapply(l: Out) = (Nil(), l)
      }

    implicit def concatCons[A, L1 <: List[A], L2 <: List[A]](implicit
        Conc: Concat[A, L1, L2]): Concat.Output[A, Cons[A, L1], L2, Cons[A, Conc.Out]] =
      new Concat[A, Cons[A, L1], L2]{
        type Out = Cons[A, Conc.Out]

        def apply(l1 : Cons[A, L1], l2 : L2) =
          Cons(l1.head, Conc(l1.tail, l2))

        def unapply(l: Cons[A, Conc.Out]) =
          Conc.unapply(l.tail) match {
            case (l1, l2) => (Cons(l.head, l1), l2)
          }
      }

  }

  class Leafs[A]{

    def get[In <: Tree[A]](t : In)(implicit S: LeafsShape[In]): S.Out = S.get(t)

    def update[In <: Tree[A]](t : In)(implicit S: LeafsShape[In]): S.Out => In = S.update(t)

    trait LeafsShape[In <: Tree[A]]{
      type Out <: List[A]

      def get(t: In): Out
      def update(t: In): Out => In
    }

    object LeafsShape{
      type Output[T <: Tree[A], _Out] = LeafsShape[T]{ type Out = _Out }

      implicit def leafCase: LeafsShape.Output[Leaf[A], Cons[A, Nil[A]]] =
        new LeafsShape[Leaf[A]]{
          type Out = Cons[A, Nil[A]]

          def get(t: Leaf[A]) =
            Cons(t.value, Nil())

          def update(t: Leaf[A]) = {
            case Cons(value, _) => Leaf(value)
          }
        }

      implicit def nodeCase[
          L <: Tree[A],
          LOut <: List[A],
          R <: Tree[A],
          ROut <: List[A]](implicit
          LeafsShapeL: LeafsShape.Output[L, LOut],
          LeafsShapeR: LeafsShape.Output[R, ROut],
          Conc: Concat[A, LOut, ROut]): LeafsShape.Output[Node[L, A, R], Conc.Out] =
        new LeafsShape[Node[L, A, R]]{
          type Out = Conc.Out

          def get(t: Node[L, A, R]) =
            Conc(LeafsShapeL.get(t.left), LeafsShapeR.get(t.right))

          def update(t: Node[L, A, R]) = Conc.unapply(_) match {
            case (il, ir) =>
              Node(LeafsShapeL.update(t.left)(il), t.root, LeafsShapeR.update(t.right)(ir))
          }
        }
    }
  }

  val leafsInt = new Leafs[Int]

  describe("Ok"){

    it("one focus"){
      val t = Leaf(1)

      leafsInt.get(t) shouldBe
        Cons(1, Nil[Int]())

      leafsInt.update(t).apply(Cons(3, Nil[Int]())) shouldBe
        Leaf(3)
    }

    it("two foci"){
      val t = Node(Node(Leaf(1), 2, Leaf(3)), 4, Leaf(5))

      leafsInt.get(t) shouldBe
        Cons(1, Cons(3, Cons(5, Nil[Int]())))

      leafsInt.update(t).apply(Cons(5, Cons(3, Cons(1, Nil[Int]())))) shouldBe
        Node(Node(Leaf(5), 2, Leaf(3)), 4, Leaf(1))
    }
  }

  describe("No OK"){

    val t = Node(Node(Leaf(1), 2, Leaf(3)), 4, Leaf(5))

    it("Not enough values"){
      """
        leafsInt.update(t).apply(Cons(1, Cons(2, Nil[Int]())))
      """ shouldNot compile
    }

    it("Exceeds values"){
      """
        leafsInt.update(t).apply(Cons(5, Cons(4, Cons(3, Cons(2, Nil[Int]())))))
      """ shouldNot compile
    }
  }
}

object TheSolutionInScalaWithNat extends TheSolutionInScalaWithNat
class TheSolutionInScalaWithNat extends FunSpec with Matchers{

  sealed abstract class Tree[A]
  case class Leaf[A](value: A) extends Tree[A]
  case class Node[L <: Tree[A], A, R <: Tree[A]](left: L, root: A, right: R) extends Tree[A]

  import shapeless.{Sized, Nat}

  class Leafs[A]{

    def get[T <: Tree[A]](t : T)(implicit E: LeafsShape[T]) = E.get(t)

    def put[T <: Tree[A]](t : T)(implicit E: LeafsShape[T]) = E.put(t)

    trait LeafsShape[T <: Tree[A]]{
      type N <: Nat

      def get(t: T): Sized[List[A], N]
      def put(t: T): Sized[List[A], N] => T
    }

    object LeafsShape{
      import shapeless.{Succ, _0}
      import shapeless.syntax.sized._

      implicit def leafCase = new LeafsShape[Leaf[A]]{
        type N = Succ[_0]

        def get(t: Leaf[A]) =
          Sized[List](t.value)

        def put(t: Leaf[A]) = {
          case Sized(value) => Leaf(value)
        }
      }

      import shapeless.ops.nat.{ToInt, Diff, Sum}

      implicit def nodeCase[
          L <: Tree[A], NL <: Nat,
          R <: Tree[A], NR <: Nat,
          M <: Nat](implicit
          LeafsShapeL: LeafsShape[L]{ type N = NL },
          LeafsShapeR: LeafsShape[R]{ type N = NR },
          S: Sum.Aux[NL, NR, M],
          D: Diff.Aux[M, NL, NR],
          TI: ToInt[NL]) = new LeafsShape[Node[L, A, R]]{
        type N = M

        def get(t: Node[L, A, R]) =
          LeafsShapeL.get(t.left) ++ LeafsShapeR.get(t.right)

        def put(t: Node[L, A, R]) = _.splitAt[NL] match {
          case (il, ir) =>
            Node(LeafsShapeL.put(t.left)(il), t.root, LeafsShapeR.put(t.right)(ir))
        }
      }
    }
  }

  val leafsInt = new Leafs[Int]

  describe("Ok"){

    it("one focus"){
      val t = Leaf(1)

      leafsInt.get(t) shouldBe
        Sized(1)

      leafsInt.put(t).apply(Sized[List](3)) shouldBe
        Leaf(3)
    }

    it("two foci"){
      val t = Node(Node(Leaf(1), 2, Leaf(3)), 4, Leaf(5))

      leafsInt.get(t) shouldBe
        Sized(1, 3, 5)

      leafsInt.put(t).apply(Sized[List](5, 3, 1)) shouldBe
        Node(Node(Leaf(5), 2, Leaf(3)), 4, Leaf(1))
    }
  }

  describe("No OK"){

    val t = Node(Node(Leaf(1), 2, Leaf(3)), 4, Leaf(5))

    it("Not enough values"){
      """
        leafsInt.put(t).apply(Sized[List](1, 2))
      """ shouldNot compile
    }

    it("Exceeds values"){
      """
        leafsInt.put(t).apply(Sized[List](1, 2, 3, 4))
      """ shouldNot compile
    }
  }

}



package org.hablapps

package object shapeaware{

  import cats.FunctorFilter, cats.data.StateT, cats.Functor, cats.Monad

  implicit def stateTFunctorFilter[F[_], S](
      implicit F: Monad[F],
      FF: FunctorFilter[F]): FunctorFilter[StateT[F, S, ?]] =
    new FunctorFilter[StateT[F, S, ?]] {
      override def functor: Functor[StateT[F, S, ?]] = Functor[StateT[F, S, ?]]

      override def mapFilter[A, B](fa: StateT[F, S, A])(
          f: A => Option[B]): StateT[F, S, B] =
        fa.flatMapF(a => FF.mapFilter(F.pure(a))(f))
    }

  import cats.syntax.functorFilter._

  implicit class WithF[F[_]: FunctorFilter, A](m: F[A]){
    def withFilter(p: A => Boolean): F[A] = m.filter(p)
  }
}

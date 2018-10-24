name := "shapeaware"

scalaVersion := "2.12.7"

scalaBinaryVersion := "2.12"

organization := "org.hablapps"

addCompilerPlugin("org.spire-math" %% "kind-projector" % "0.9.3")

libraryDependencies ++= Seq(
  "org.scalatest" %% "scalatest" % "3.0.0",
  "com.chuusai" %% "shapeless" % "2.3.3",
  "org.typelevel" %% "cats-core" % "1.4.0"
)


scalacOptions ++= Seq(
  "-unchecked",
  "-deprecation",
  "-Ypartial-unification",
  "-explaintypes",
  "-feature",
  "-language:implicitConversions",
  "-language:postfixOps",
  "-language:higherKinds")

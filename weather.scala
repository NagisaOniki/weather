//report6
//気象画像(H:581×W:600)から低気圧をよみとく

import breeze.linalg._
import CLASS._
import scala.sys.process._
import scala.language.postfixOps

object weather{

  val dn = 100
  val ln = 1000
  val rand = new scala.util.Random(0)
  val H = 194
  val W = 200

  ////////////////////低気圧の位置探し/////////////////////////////////////
  def create_answer(V:Array[Double],H:Int,W:Int)={
    def iindex(i:Int, j:Int,c:Int) = i * W * 3 + j * 3 + c
    val BW = 24 //低の縦（横）
    val Z = Array.ofDim[Double](3 * H * W)
    for(i <- 0 until H-BW; j <- 0 until W-BW ){ //; k<-0 until 3) { //画素値の合計
      var R = 0d
      var G = 0d
      var B = 0d
      for(m <- 0 until BW; n <- 0 until BW) {
        R += V(iindex(i+m,j+n,0))*256
        G += V(iindex(i+m,j+n,1))*256
        B += V(iindex(i+m,j+n,2))*256
      }
      if(R/(BW*BW) > 200 && G/(BW*BW) < 200 && B/(BW*BW) < 200){
        for(m <-0 until BW ; n <-0 until BW){
          Z(iindex(i+m,j+n,0)) = 1 //低の位置を白で塗りつぶす
          Z(iindex(i+m,j+n,1)) = 1 //低の位置を白で塗りつぶす
          Z(iindex(i+m,j+n,2)) = 1 //低の位置を白で塗りつぶす
        }
      }
    }
    Z
  }
  ///////////////////教師データ作成///////////////////////////////////////
  def create_dtrain(){
    //------ディレクトリ内のファイル名を改行で読み込み（!!:pにファイル名格納させる)-------------
    val p = (Process("ls /home/share/chijou/") !!).split("\n")
    //------jp_cを含むファイル名格納-------------
    val file = p.filter(_.contains("jp_c"))
    //-----教師データ読み込み->pngからtxtに変換-------
    for(i<-0 until dn){
      println("convert:"+i+"回目")
      var before = "/home/share/chijou/"+file(i)
      var after = "weatherTrain/weather"+i+".txt"
      val p = Process("ipython src/main/python/convert_c.py " + before + " " + after).run
      p.exitValue
    }
    //------教師データ画像化-------
    val dtrain0 =
      (for(i<-0 until dn) yield {
        println("createAnswer:"+i+"回目")
        def fd(line:String) = line.split(",").map(_.toDouble / 256).toArray
        val train_d = scala.io.Source.fromFile("weatherTrain/weather"+i+".txt").getLines.map(fd).toArray
        var im = create_answer(train_d.flatten,581,600) //縦581横600 //低の位置指定
        var im2 = Image.to3DArrayOfColor(Image.encode(im),581,600) //RGB順に並び替え -> 一次元を三次元に
        Image.write(f"weatherTrain/teacher"+i+".png" , im2) //画像化
      })
  }


  //////////////////////main/////////////////////////////////////////////
  def main(args:Array[String]){

  /*  ////リサイズせずに使用/////
    //----正解データ読み込み-----------------
    val p = (Process("ls /home/share/chijou/") !!).split("\n")
     val file = p.filter(_.contains("jp_c"))
     val train_d =
     (for(i<-0 until dn) yield {
     Image.encode3D((Image.read("/home/share/chijou/"+file(i))).map(_.map(_.map(_.toDouble))))
     })
    //----教師データ読み込み----------
    val train_t =
     (for(i<-0 until dn) yield {
     Image.encode3D((Image.read("weatherTrain/teacher"+i+".png")).map(_.map(_.map(_.toDouble))))
     })
    
*/

    /////リサイズして使用//////
    //-----正解データリサイズ--------
    val p = (Process("ls /home/share/chijou/") !!).split("\n")
    val file2 = p.filter(_.contains("jp_c"))
    for(i<-0 until dn){
      var before = "/home/share/chijou/"+file2(i)
      var after = "weatherTrain/chijou"+i+".png"
      val p = Process("convert -geometry 200x200 " + before + " " + after).run
      p.exitValue
    }
    //-----教師データリサイズ----------
    for(i<-0 until dn){
      var before = "weatherTrain/teacher"+i+".png"
      var after = "weatherTrain/teacherResize"+i+".png"
      val p = Process("convert -geometry 200x200 " + before + " " + after).run
      p.exitValue
    }
    //----正解データ読み込み-----------------
     val train_d =
       (for(i<-0 until dn) yield {
         val xs =  (Image.read("weatherTrain/chijou"+i+".png")).map(_.map(_.map(_.toDouble)))
         Image.encode3D(xs)
       })
    //----教師データ読み込み----------
    val train_t =
      (for(i<-0 until dn) yield {
        val xs = (Image.read("weatherTrain/teacherResize"+i+".png")).map(_.map(_.map(_.toDouble)))
        Image.encode3D(xs)
      })


    ///////データ作成/////////
    //-----入力データ作成------------
    val dtrain = train_d.zip(train_t)
    //----入力データまとめ------------
    //val ds = (0 until dn).map(i=>dtrain(i)._1).toArray

    //-----ネットワーク構成------------------
    val N = new network()
    //CNN
   /* val Layer = List(
      new Convolution(3,581,600,1,20),
      new ReLU(),
      new Pooling(2,20,26,26),
      new Convolution(4,13,13,20,20),
      new ReLU(),
      new Pooling(2,20,10,10),
      new Affine(5*5*20,10)
    )*/
    val Layer = List(
      new Affine(H*W*3,100),
      new ReLU(),
      new Affine(100,H*W*3)
    )
/*
    //--------Convolution初期化-------------
    for(i<-0 until Layer(0).OC ; j<-0 until Layer(0).IC ; k<-0 until Layer(0).KW ; l<-0 until Layer(0).KW){
      Layer(0).K(i)(j)(k)(l) = rand.nextDouble * 0.01
    }
    for(i<-0 until Layer(3).OC ; j<-0 until Layer(3).IC ; k<-0 until Layer(3).KW ; l<-0 until Layer(3).KW){
      Layer(3).K(i)(j)(k)(l) = rand.nextDouble * 0.01
    }
 */

    //-------学習-------------
    for(i<-0 until ln){
      println("3:"+i+"回目")
      var correct = 0d
      var err = 0d
      var j = 0d //画像確認
      for((x,n)<-dtrain.take(dn)){
        j += 1
        val y = N.forwards(Layer,x) //y:Array[Double]
        var d = new Array[Double](H*W*3)
        for(k<-0 until d.size){
          d(k) = y(k) - n(k)
        }
        N.backwards(Layer.reverse,d)
        N.updates(Layer)
        N.resets(Layer)    
        if((i==0 || i==50 || i==100 || i==200 || i==300 || i==400 || i%500==0 || i==ln-1)
        && j%10==0){
          Image.write(f"weather3-"+i+"-"+j+".png" , Image.make_Wimage(y,1,1,H,W))
        }
      }//take

    }//ln
  }//main
}//object

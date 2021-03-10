// 对应无序列表的那行字
var laopo1_text = document.getElementById("laopo1")
var laopo2_text = document.getElementById("laopo2")
var laopo3_text = document.getElementById("laopo3")

// 点击文本就执行函数
laopo1_text.addEventListener("click", picLink);
laopo2_text.addEventListener("click", picLink);
laopo3_text.addEventListener("click", picLink);


function picLink() {
    // 先把所有图片隐藏
    var allImages = document.querySelectorAll("img");

    for (var i=0; i<allImages.length; i++) {
        allImages[i].className = "hide";
    }
    // 由自定义属性拿到图片的ID,this表示当前点击的对象
    var picId = this.attributes["data-img"].value;
    var pic = document.getElementById(picId);
    // 改变图片对象的className,显示图片
    pic.className = "";
}


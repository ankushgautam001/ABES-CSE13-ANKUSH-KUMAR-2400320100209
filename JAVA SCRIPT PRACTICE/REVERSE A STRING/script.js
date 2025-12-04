function reverseText(){
    let val=document.getElementById("box").value;
    let rev=val.split("").reverse().join("");
    document.getElementById("output").innerHTML="Reversed: " + rev;
}
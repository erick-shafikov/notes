<!DOCTYPE html>
<html>
<body>
  
  <script>

    function func(n) {

      let arr = new Array(n);

      for (let i = 0; i < n ; i++){

        for (let j = 0; j < n; j++ ){
          if (n - j - 1 > i){
            arr[j] = "X";
          } else {
            arr[j] = "Y"
          }

        }

        console.log( arr.join(" "));

      
    }
  }

  func(6);


    
  </script>

</body>

</html>
void name_0 ( char * * dataPtr ) { char  * data  =  *  dataPtr   ; { size_t i , destLen ; char dest [ 100 ] ;  memset  ( dest 'C'  100  +  1  ) ;  dest [  100  +  1  ]  =  '\0'  ;  destLen  =   strlen  ( dest )  ; for (  i  =  0  ;  i  <  destLen   i  ++ ) {  dest [ i ]  =  data [ i ]  ; }  dest [  100  +  1  ]  =  '\0'  ;  printLine  ( dest ) ; } } void name_1 ( ) { char * data ; char  * dataBadBuffer  =  ( char * )   ALLOCA  (  50  *  sizeof ( char )  )   ; char  * dataGoodBuffer  =  ( char * )   ALLOCA  (  100  *  sizeof ( char )  )   ;  memset  ( dataBadBuffer 'A'  50  +  1  ) ;  dataBadBuffer [  50  +  1  ]  =  '\0'  ;  memset  ( dataGoodBuffer 'A'  100  +  1  ) ;  dataGoodBuffer [  100  +  1  ]  =  '\0'  ;  data  =   dataGoodBuffer  +  8   ;  name_0  ( &  data  ) ; } int main ( int argc char * argv [ ] ) {  srand  ( ( unsigned )   time  ( NULL )  ) ;  name_1  ( ) ; return 0 ; } 
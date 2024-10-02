-- Parallel Longest Satisfying Segment
--
-- ==
-- compiled input {
--    [1, -2, -2, 0, 0, 0, 0, 0, 3, 4, -6, 1]
-- }  
-- output { 
--    9
-- }
-- compiled input {
--    [1i32, -2i32, -2i32, 0i32, 0i32, 0i32, 0i32, 3i32, 4i32, -6i32, 1i32, 0i32, 0i32, 0i32, 0i32]
-- }
-- output {
--    8i32
-- }
-- compiled input {
--    [1i32]
-- }
-- output {
--    1i32
-- }
-- compiled input {
--    empty([0]i32)
-- }
-- output {
--    0i32
-- }

import "lssp"
import "lssp-seq"

type int = i32

let main (xs: []int) : int =
  let pred1 _   = true
  let pred2 x y = (x <= y)
  in  lssp_seq pred1 pred2 xs
--  in  lssp pred1 pred2 xs

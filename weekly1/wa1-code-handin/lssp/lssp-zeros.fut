-- Parallel Longest Satisfying Segment
--
-- ==
-- compiled input {
--    [1i32, -2, -2, 0, 0, 0, 0, 0, 3, 4, -6, 1]
-- }
-- output {
--    5
-- }
-- compiled input {
--    [1i32, -2i32, -2i32, 0i32, 0i32, 0i32, 0i32, 3i32, 4i32, -6i32, 1i32, 0i32, 0i32, 0i32, 0i32]
-- }
-- output {
--    4i32
-- }
-- compiled input {
--    [0i32]
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

import "lssp-seq"
import "lssp"

type int = i32

let main (xs: []int) : int =
  let pred1 x   = (x == 0)
  let pred2 x y = (x == 0) && (y == 0)
  in  lssp_seq pred1 pred2 xs
--  in  lssp pred1 pred2 xs

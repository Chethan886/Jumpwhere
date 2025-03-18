#217. Contains Duplicate
class Solution:
    def containsDuplicate(self, nums: List[int]) -> bool:
        return len(set(nums))!=len(nums)
    
#268. Missing Number
class Solution:
    def missingNumber(self, nums: List[int]) -> int:
        n=len(nums)
        sum1=(n*(n+1))//2
        sum2=sum(nums)
        return sum1-sum2
    
#448. Find All Numbers Disappeared in an Array
class Solution:
    def findDisappearedNumbers(self, nums: List[int]) -> List[int]:
        ss=set(nums)
        arr=[]
        for i in range(1,len(nums)+1):
            if i not in ss:
                arr.append(i)
        return arr
    
#1. Two Sum
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        ss={}
        for i,x in enumerate(nums):
            if target-x in ss:
                return i,ss[target-x]
            else:
                ss[x]=i


#1365 How Many Numbers Are Smaller Than the Current Number
class Solution:
    def smallerNumbersThanCurrent(self, nums: List[int]) -> List[int]:
        hash_map={}
        arr=sorted(nums)
        for i,x in enumerate(arr):
            if x in hash_map:
                continue
            hash_map[x]=i
        t=[]
        for a in nums:
            t.append(hash_map[a])
        return t
    
#1266. Minimum Time Visiting All Points
class Solution:
    def minTimeToVisitAllPoints(self, points: List[List[int]]) -> int:
        res=0
        x1,y1=points.pop()
        while points:
            x2,y2=points.pop()
            res+=max(abs(y2-y1),abs(x2-x1))
            x1,y1=x2,y2
        return res
    
#54. Spiral Matrix
class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        ret=[]
        while matrix:
            ret+=(matrix.pop(0))

            if matrix and matrix[0]:
                for row in matrix:
                    ret.append(row.pop())

            if matrix:
                ret+=(matrix.pop()[::-1])       

            if matrix and matrix[0]:
                for row in matrix[::-1]:
                    ret.append(row.pop(0))
        return ret 

#200. Number of Islands
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        if not grid:
            return 0
        def bfs(r,c):
            q=deque()
            visit.add((r,c))
            q.append((r,c))

            while q:
                row,col=q.popleft()
                directions=[[1,0],[-1,0],[0,1],[0,-1]]
                for dr,dc in directions:
                    r,c=row+dr,col+dc
                    if (r in range(rows) and c in range(cols) and grid[r][c]=='1' and (r,c) not in visit):
                        q.append((r,c))
                        visit.add((r,c))

        count=0
        rows=len(grid)
        cols=len(grid[0])
        visit=set()
        for r in range(rows):
            for c in range(cols):
                if grid[r][c]=='1' and (r,c) not in visit:
                    bfs(r,c)
                    count+=1
        return count
        
#121. Best Time to Buy and Sell Stock
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        l,r=0,1
        maxP=0
        while r!=len(prices):
            if prices[l]<prices[r]:
                prof=prices[r]-prices[l]
                maxP=max(maxP,prof)
            else:
                l=r
            r+=1
        return maxP

#977. Squares of a Sorted Array
class Solution:
    def sortedSquares(self, nums: List[int]) -> List[int]:
        arr=[x*x for x in nums]
        res=sorted(arr)
        return res
 #or
class Solution:
    def sortedSquares(self, nums: List[int]) -> List[int]:
        answer=collections.deque()
        l,r=0,len(nums)-1
        while l<=r:
            left,right = abs(nums[l]),abs(nums[r])
            if left>right:
                answer.appendleft(left*left)
                l+=1
            else:
                answer.appendleft(right*right)
                r-=1
        return list(answer)

#15. 3Sum
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        triplets=[]
        nums.sort()
        for indx,val in enumerate(nums):
            if (indx>0) & (val==nums[indx-1]):
                continue
        
            left=(indx+1)
            right= len(nums)-1

            while left<right:
                curSum=val+nums[left]+nums[right]
                if curSum>0:
                    right-=1
                elif curSum<0:
                    left+=1
                else:
                    triplets.append([val,nums[left],nums[right]])
                    left+=1

                    while (left<right) & (nums[left]==nums[left-1]):
                        left+=1
        return triplets

#845. Longest Mountain in Array
class Solution:
    def longestMountain(self, arr: List[int]) -> int:
        m=0
        for i in range(1,len(arr)-1):
            if arr[i-1]<arr[i]>arr[i+1]:
                lc,rc=1,1
                l,r=i-1,i+1
                while l>0:
                    if arr[l-1]<arr[l]:
                        lc+=1
                        l-=1
                    else:
                        break
                while r<len(arr)-1:
                    if arr[r+1]<arr[r]:
                        rc+=1
                        r+=1
                    else:
                        break   
                m=max(m,(lc+rc+1))         
        return m
                
#219. Contains Duplicate II
class Solution:
    def containsNearbyDuplicate(self, nums: List[int], k: int) -> bool:
        seen=set()
        for i,num in enumerate(nums):
            if num in seen:
                return True
            seen.add(num)
            if len(seen)>k:
                seen.remove(nums[i-k])
        return False
        
#1200. Minimum Absolute Difference
class Solution:
    def minimumAbsDifference(self, arr: List[int]) -> List[List[int]]:
        arr.sort()
        min_dif=float('inf')
        for i in range(len(arr)-1):
            min_dif=min(min_dif,arr[i+1]-arr[i])
        res=[]
        for i in range(len(arr)-1):
            if arr[i+1]-arr[i]==min_dif:
                res.append([arr[i],arr[i+1]])
        return res

#209. Minimum Size Subarray Sum
class Solution:
    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        l=0
        total=0
        res=float('inf')
        print(res)
        for r in range(len(nums)):
            total+=nums[r]
            while total>=target:
                res=min(res,r-l+1)
                total-=nums[l]
                l+=1
        if res==float('inf'):
            return 0
        return res

#136. Single Number
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        res=nums[0]
        for i in range(1,len(nums)):
            res^=nums[i]
        return res

#Dynamic programing
#322. Coin Change
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        dp=[amount+1]*(amount+1)
        dp[0]=0
        for i in range(1,amount+1):
            for c in coins:
                if (i-c)>=0:
                    dp[i]=min(dp[i],1+dp[i-c])
        return dp[amount] if (dp[amount ]!=amount+1) else -1

#70. Climbing Stairs
class Solution:
    def climbStairs(self, n: int) -> int:
        if n==1:
            return 1
        dp=[0]*(n+1)
        dp[0]=0
        dp[1]=1
        dp[2]=2
        for i in range(3,n+1):
            dp[i]=dp[i-1]+dp[i-2]
        return dp[n]

#53. Maximum Subarray
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        dp=[0]*len(nums)
        for i,n in enumerate(nums):
            dp[i]=max(n,dp[i-1]+n)
        return max(dp)

#338. Counting Bits
class Solution:
    def countBits(self, n: int) -> List[int]:
        dp=[0]*(n+1)
        offset=1
        for i in range(1,n+1):
            if offset*2==i:
                offset=i
            dp[i]=1+dp[i-offset]
        return dp

#303. Range Sum Query - Immutable
class NumArray:

    def __init__(self, nums: List[int]):
        self.nums=nums

    def sumRange(self, left: int, right: int) -> int:
        return sum(self.nums[left:right+1])
#or (dynamic programing)
class NumArray:

    def __init__(self, nums: List[int]):
        self.acc_nums=[0]
        for num in nums:
            self.acc_nums.append(self.acc_nums[-1]+num)

    def sumRange(self, left: int, right: int) -> int:
        return self.acc_nums[right+1]-self.acc_nums[left]

#Backtracking
#784. Letter Case Permutation
class Solution:
    def letterCasePermutation(self, s: str) -> List[str]:
        res=[]
        def backtrack(sub="",i=0):
            if len(sub)==len(s):
                res.append(sub)
                return
            if s[i].isalpha():
                backtrack(sub+s[i].swapcase(),i+1)
            backtrack(sub+s[i],i+1)
        backtrack()
        return res

#78. Subsets
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        def backtrack(start,path):
            res.append(path[:])
            for i in range(start,len(nums)):
                path.append(nums[i])
                backtrack(i+1,path)
                path.pop()
        res=[]
        backtrack(0,[])
        return res

#77. Combinations
class Solution:
    def combine(self, n: int, k: int) -> List[List[int]]:
        def backtrack(start,path):
            if len(path)==k:
                result.append(path[:])
                return
            for i in range(start,n+1):
                path.append(i)
                backtrack(i+1,path)
                path.pop()
        result=[]
        backtrack(1,[])
        return result

#44. Permutations
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        def backtrack(start,end):
            if start==end:
                result.append(nums[:])
                return
            for i in range(start,end):
                nums[start],nums[i]=nums[i],nums[start]
                backtrack(start+1,end)
                nums[start],nums[i]=nums[i],nums[start]
        result=[]
        backtrack(0,len(nums))
        return result
    
#Linked List
#876. Middle of the Linked List
class Solution:
    def middleNode(self, head: Optional[ListNode]) -> Optional[ListNode]:
        slow=fast=head
        while fast and fast.next:
            slow=slow.next
            fast=fast.next.next
        return slow

#141. Linked List Cycle
class Solution:
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        slow,fast=head,head
        while (fast) and fast.next:
            slow=slow.next
            fast=fast.next.next
            if slow==fast:
                return True
        return False

#206. Reverse Linked List      
class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        prev=None
        curr=head
        while (curr!=None):
            next_pointer=curr.next
            curr.next=prev
            prev=curr
            curr=next_pointer
        return prev
#203. Remove Linked List Elements
class Solution:
    def removeElements(self, head: Optional[ListNode], val: int) -> Optional[ListNode]:
        if not head:
            return head
        while head.val==val:
            if head.next:
                head=head.next
            else:
                head=None
                return head
        curr=head.next
        prev=head
        while curr!=None:
            if curr.val==val:
                if curr.next==None:
                    prev.next=None
                prev.next=curr.next
                curr=curr.next
                continue
            curr=curr.next
            prev=prev.next
        return head
#or
class Solution:
    def removeElements(self, head: Optional[ListNode], val: int) -> Optional[ListNode]:
        dummy=ListNode(-1)
        dummy.next=head
        curr=dummy
        while curr.next!=None:
            if curr.next.val==val:
                curr.next=curr.next.next
            else:
                curr=curr.next
        return dummy.next

#92. Reverse Linked List II
class Solution:
    def reverseBetween(self, head: Optional[ListNode], left: int, right: int) -> Optional[ListNode]:
        dummy=ListNode(-1,head)
        left_prev,curr=dummy,head
        for i in range(left-1):
            left_prev,curr=curr,curr.next
        prev=None
        for i in range(right-left+1):
            next_ptr=curr.next
            curr.next=prev
            prev,curr=curr,next_ptr
        left_prev.next.next=curr
        left_prev.next=prev
        return dummy.next

#234. Palindrome Linked List
class Solution:
    def isPalindrome(self, head: Optional[ListNode]) -> bool:
        fast,slow=head,head
        while (fast) and (fast.next):
            fast=fast.next.next
            slow=slow.next
        
        prev=None
        while slow!=None:
            n_ptr=slow.next
            slow.next=prev
            prev=slow
            slow=n_ptr
        
        left=head
        right=prev
        while right!=None:
            if left.val!=right.val:
                return False
            left=left.next
            right=right.next
        return True

#21. Merge Two Sorted Lists
class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        curr=ListNode(-1)
        dummy=curr
        while list1 and list2:
            if list1.val<list2.val:
                dummy.next=list1
                list1=list1.next
            else:
                dummy.next=list2
                list2=list2.next
            dummy=dummy.next
        if list1:
            dummy.next=list1
        if list2:
            dummy.next=list2
        return curr.next

#155. Min Stack
class MinStack:
    def __init__(self):
        self.stack=[]

    def push(self, val: int) -> None:
        if not self.stack:
            current_min=val
        else:
            current_min=min(val,self.stack[-1][1])
        self.stack.append((val,current_min))

    def pop(self) -> None:
        self.stack.pop()

    def top(self) -> int:
        return self.stack[-1][0]

    def getMin(self) -> int:
        return self.stack[-1][1]

#20. Valid Parentheses
class Solution:
    def isValid(self, s: str) -> bool:
        stack=[]
        hashmap={')':'(',   '}':'{',   ']':'['}
        for x in s:
            if stack and (x in hashmap and stack[-1]== hashmap[x]):
                stack.pop()
            else:
                stack.append(x)
        return not stack

#150. Evaluate Reverse Polish Notation
class Solution:
    def evalRPN(self, tokens: List[str]) -> int:
        stack=[]
        for t in tokens:
            if t not in "+-*/":
                stack.append(int(t))
            else:
                r,l=stack.pop(),stack.pop()
                if t=="+":
                    stack.append(l+r)
                elif t=="-":
                    stack.append(l-r)
                elif t=="*":
                    stack.append(l*r)
                else:
                    stack.append(int(float(l)/r))
        return stack.pop()

#Sorting a Stack
def sortstack(stack):
    tmpstack=[]
    while stack:
        num=stack.pop()
        while (tmpstack and tmpstack[-1]<num):
            stack.append(tmpstack.pop())
        tmpstack.append(num)
    return tmpstack

#Queue
#225 Implement Stack using Queues
class MyStack:
    def __init__(self):
        self.queue=deque()

    def push(self, x: int) -> None:
        self.queue.append(x)
        
    def pop(self) -> int:
        for i in range(len(self.queue)-1):
            self.push(self.queue.popleft())
        return self.queue.popleft()

    def top(self) -> int:
        return self.queue[-1]

    def empty(self) -> bool:
        return len(self.queue)==0

#Reverse first K elements of Queue using stack
def reverse_first_k_elements(k,q):
    stack=[]
    for i in range(k):
        stack.append(q.popleft())
    while stack:
        q.append(stack.pop())
    for i in range(len(q)-k):
        q.append(q.popleft())
    return q

#Binary Trees
#637. Average of Levels in Binary Tree
class Solution:
    def averageOfLevels(self, root: Optional[TreeNode]) -> List[float]:
        queue=deque([root])
        result=[]

        while queue:
            level=[]
            for i in range(len(queue)):
                node=queue.popleft()
                level.append(node.val)
                print(level)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            result.append(sum(level)/len(level))
        return result

#111. Minimum Depth of Binary Tree
class Solution:
    def minDepth(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0
        queue=deque([(root,1)])
        while queue:
            node,level=queue.popleft()
            if not node.left and not node.right:
                return level
            if node.left:
                queue.append((node.left,level+1))
            if node.right:
                queue.append((node.right,level+1))
        return 0

#104. Maximum Depth of Binary Tree
class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0
        queue=deque([(root,1)])
        while queue:
            node,level=queue.popleft()
            if node.right:
                queue.append((node.right,level+1))
            if node.left:
                queue.append((node.left,level+1))
        return level
    #or
class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0
        queue=deque([(root,1)])
        return  max(self.maxDepth(root.left),self.maxDepth(root.right))+1

#Max/min value of binary tree
def largest(root):
    queue=deque([root])
    max_node=0
    while queue:
        curr_node=queue.popleft()
        if curr_node.left:
            queue.append(curr_node.left)
        if curr_node.right:
            queue.append(curr_node.right)
        if curr_node.val>max_node:
            max_node=curr_node.val
    return max_node

#102. Binary Tree Level Order Traversal
class Solution:
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if root==None:
            return []
        queue=deque([root])
        tree=[]
        while queue:
            level=[]
            for i in range(len(queue)):
                node =queue.popleft()
                level.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            tree.append(level)
        return tree

#100. Same Tree
class Solution:
    def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
        stack=[(p,q)]
        while stack:
            node1,node2=stack.pop()
            if not node1 and not node2:
                continue
            elif None in [node1,node2] or node1.val!= node2.val:
                return False
            stack.append((node1.right,node2.right))
            stack.append((node1.left,node2.left))
        return True

#112. Path Sum
class Solution:
    def hasPathSum(self, root: Optional[TreeNode], targetSum: int) -> bool:
        if not root:
            return False
        stack=[(root,root.val)]
        while stack:
            curr,val=stack.pop()
            if not curr.left and not curr.right and val==targetSum:
                return True
            if curr.right:
                stack.append((curr.right,val+curr.right.val))
            if curr.left:
                stack.append((curr.left,val+curr.left.val))

        return False

#543. Diameter of Binary Tree
class Solution:
    def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        self.diameter=0
        def depth(root):
            if not root:
                return 0
            left_depth=depth(root.left)
            right_depth=depth(root.right)

            self.diameter=max(self.diameter,left_depth+right_depth)
            return 1+max(left_depth,right_depth)
        depth(root)
        return self.diameter
    
#226. Invert Binary Tree
class Solution:
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        stack=[root]
        while stack:
            curr=stack.pop()
            if curr:
                curr.left,curr.right=curr.right,curr.left
                stack.extend([curr.right,curr.left])
        return root

#236. Lowest Common Ancestor of a Binary Tree
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        queue=deque([root])
        parent={root:None}
        while queue:
            node=queue.popleft()
            if node.left:
                queue.append(node.left)
                parent[node.left]=node
            if node.right:
                queue.append(node.right)
                parent[node.right]=node
            if p in parent and q in parent:
                break
        ancestors=set()
        while p:
            ancestors.add(p)
            p=parent[p]
        while q:
            if q in ancestors:
                return q
            q=parent[q]

#700. Search in a Binary Search Tree
class Solution:
    def searchBST(self, root: Optional[TreeNode], val: int) -> Optional[TreeNode]:
        while root:
            if root.val==val:
                return root
            elif root.val<val:
                root=root.right
            else:
                root=root.left
        return None

#701. Insert into a Binary Search Tree
class Solution:
    def insertIntoBST(self, root: Optional[TreeNode], val: int) -> Optional[TreeNode]:
        new_node=TreeNode(val)
        if not root:
            return new_node
        curr=root
        while True:
            if val<curr.val:
                if curr.left:
                    curr=curr.left
                else:
                    curr.left=new_node
                    break
            else:
                if curr.right:
                    curr=curr.right
                else:
                    curr.right=new_node
                    break
        return root

#108. Convert Sorted Array to Binary Search Tree
class Solution:
    def sortedArrayToBST(self, nums: List[int]) -> Optional[TreeNode]:
        if not nums:
            return None
        mid=len(nums)//2
        root=TreeNode(nums[mid])
        root.left=self.sortedArrayToBST(nums[:mid])
        root.right=self.sortedArrayToBST(nums[mid+1:])
        return root

#653. Two Sum IV - Input is a BST
class Solution:
    def findTarget(self, root: Optional[TreeNode], k: int) -> bool:
        queue=deque([root])
        num=set()
        while queue:
            node=queue.popleft()
            if (k-node.val) in num:
                return True
            else:
                num.add(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
        return False

#235. Lowest Common Ancestor of a Binary Search Tree
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        small=min(p.val,q.val)
        large=max(p.val,q.val)
        while root:
            if root.val>large:
                root=root.left
            elif root.val<small:
                root=root.right
            else:
                return root
        return None

#530. Minimum Absolute Difference in BST
class Solution:
    def getMinimumDifference(self, root: Optional[TreeNode]) -> int:
        mindiff=float('inf')
        prev_val=float('-inf')
        stack=[]
        while root or stack:
            if root:
                stack.append(root)
                root=root.left
            else:
                root=stack.pop()
                mindiff=min(mindiff,root.val-prev_val)
                prev_val=root.val
                root=root.right
        return mindiff

#1382. Balance a Binary Search Tree
class Solution:
    def balanceBST(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        def inorder_traversal(node: Optional[TreeNode]) -> List[int]:
            if not node:
                return []
            return inorder_traversal(node.left) + [node.val] + inorder_traversal(node.right)

        def sorted_list_to_bst(start: int, end: int) -> Optional[TreeNode]:
            if start > end:
                return None
            mid = (start + end) // 2
            root = TreeNode(values[mid])  
            root.left = sorted_list_to_bst(start, mid - 1) 
            root.right = sorted_list_to_bst(mid + 1, end)  
            return root

        values = inorder_traversal(root)

        return sorted_list_to_bst(0, len(values) - 1)

#450. Delete Node in a BST
class Solution:
     def deleteNode(self, root: Optional[TreeNode], key: int) -> Optional[TreeNode]:
        if not root:
            return None
        
        parent = None
        current = root

        while current and current.val != key:
            parent = current
            if key < current.val:
                current = current.left
            else:
                current = current.right

        if not current:
            return root

        if not current.left and not current.right:
            if not parent:
                return None 
            if parent.left == current:
                parent.left = None
            else:
                parent.right = None

        elif not current.left or not current.right:
            child = current.left if current.left else current.right
            if not parent:
                return child  
            if parent.left == current:
                parent.left = child
            else:
                parent.right = child

        else:
            successor_parent = current
            successor = current.right
            while successor.left:
                successor_parent = successor
                successor = successor.left
            
            current.val = successor.val
            if successor_parent.left == successor:
                successor_parent.left = successor.right
            else:
                successor_parent.right = successor.right
        
        return root

#230. Kth Smallest Element in a BST
class Solution:
    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        stack = []
        while root or stack:
            while root:
                stack.append(root)
                root = root.left
            
            root = stack.pop()
            k -= 1
            if k == 0:
                return root.val 
            
            root = root.right 

#Heap
#215. Kth Largest Element in an Array
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        return heapq.nlargest(k,nums)[-1]
#or
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        heap=[]
        for i in nums:
            heapq.heappush(heap,i)
        for i in range(len(nums)-k):
            heapq.heappop(heap)
        return heapq.heappop(heap)

#973. K Closest Points to Origin
class Solution:
    def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
        heap=[]
        for (x,y) in points:
            dist=-(x*x+y*y)
            if len(heap)==k:
                heapq.heappushpop(heap,(dist,x,y))
            else:
                heapq.heappush(heap,(dist,x,y))
        return [(x,y) for (dist,x,y) in heap]

#347. Top K Frequent Elements
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        count=Counter(nums)
        heap=[]
        for num,freq in count.items():
            if len(heap)<k:
                heapq.heappush(heap,(freq,num))
            elif freq>heap[0][0]:
                heapq.heapreplace(heap,(freq,num))
        top_k=[num for freq,num in heap]
        return top_k

#621. Task Scheduler
class Solution:
    def leastInterval(self, tasks: List[str], n: int) -> int:
        counts=Counter(tasks)
        heap=[]
        for count in counts.values():
            heap.append(-count)
        heapq.heapify(heap)
        time=0
        wait=deque()
        while heap or wait:
            time+=1
            if heap:
                cur=heapq.heappop(heap)
                cur+=1
                if cur!=0:
                    wait.append((cur,time+n))
            if wait and wait[0][1]==time:
                heapq.heappush(heap,wait.popleft()[0])
        return time
    
#Graph
#133. Clone Graph
class Solution:
    def cloneGraph(self, node: Optional['Node']) -> Optional['Node']:
        if not node:
            return node
        queue=deque([node])
        clones={node.val:Node(node.val)}
        while queue:
            curr=queue.popleft()
            curr_clone=clones[curr.val]

            for neighbor in curr.neighbors:
                if neighbor.val not in clones:
                    clones[neighbor.val]=Node(neighbor.val)
                    queue.append(neighbor)
                curr_clone.neighbors.append(clones[neighbor.val])
        return clones[node.val]

#787. Cheapest Flights Within K Stops
class Solution:
    def findCheapestPrice(self, n: int, flights: List[List[int]], src: int, dst: int, k: int) -> int:
        prices = [float("inf")] * n
        prices[src] = 0

        for i in range(k + 1):
            tmpPrices = prices.copy()

            for from_node, to_node, cost in flights:
                if prices[from_node] == float("inf"):
                    continue
                if prices[from_node] + cost < tmpPrices[to_node]:
                    tmpPrices[to_node] = prices[from_node] + cost

            prices = tmpPrices

        if prices[dst] == float("inf"):
            return -1
        else:
            return prices[dst]

#207. Course Schedule
class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        adj = {course: [] for course in range(numCourses)}
        for course, pre in prerequisites:
            adj[course].append(pre)

        for course in range(numCourses):
            stack = [(course, set())]
            while stack:
                cur_course, visited = stack.pop()
                if cur_course in visited:
                    return False  
                visited.add(cur_course)
                for pre in adj[cur_course]:
                    stack.append((pre, visited.copy()))
            adj[course] = []  

        return True
        
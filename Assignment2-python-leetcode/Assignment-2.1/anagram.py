#anagrams
def anagrams(s1,s2):
    if len(s1)==len(s2):
        return False
    f1={}
    f2={}
    for ch in s1:
        if ch in f1:
            f1[ch]+=1
        else:
            f1[ch]=1
    for ch in s2:
        if ch in f2:
            f2[ch]+=1
        else:
            f2[ch]=1
    for key in f1:
        if key not in f2 or f1[key]!=f2[key]:
            return False
    return True

#first and last positions
def position(arr,target):
    for i in range(len(arr)):
        if arr[i]==target:
            start=i
            while i+1 <len(arr) and arr[i+1]==target:
                i+=1
            return [start,i]
    return [-1,-1]

#kth largest
def k_largest(arr,k):
    for i in range(k-1):
        arr.remove(max(arr))
    return max(arr)
def k_largest(arr,k):
    n=len(arr)
    arr.sort()
    return arr[n-k]


#symmetric trees
def are_symmetric(r1,r2):
    if r1 is None and r2 is None:
        return True
    elif ((r1 is None))!=((r2 is None)) or r1.val!=r2.val:
        return False
    else:
        return are_symmetric(r1.left,r2.right) and (r1.right,r2.left)
def is_symmetric(root):
    if root is None:
        return True
    return are_symmetric(root.left,root.right)


#general paranthesis
def is_valid(combination):
    stack=[]
    for par in combination:
        if par=='(':
            stack.append(par)
        else:
            if len(stack)==0:
                return False
            else:
                stack.pop()
    return len(stack)==0

def generate(n):
    def rec(n,diff,comb,combs):
        if diff<0:
            return
        elif n==0:
            if diff==0:
                combs.append(''.join(comb))
        else:
            comb.append('(')
            rec(n-1,diff+1,comb,combs)
            comb.pop()
            comb.append(')')
            rec(n-1,diff-1,comb,combs)
            comb.pop()
    combs=[]
    rec(2*n,0,[],combs)
    return combs


#circular gas station
def can_traverse(gas,cost,start):
    n=len(gas)
    remaining=0
    i=start
    started=False
    while i!= start or not started:
        started=True
        remaining+=gas[i]-cost[i]
        if remaining<0:
            return False
        i=(i+1)%n
    return True
def gas_station(gas,cost):
    for i in range(len(gas)):
        if can_traverse(gas,cost,i):
            return i
    return -1

#optimized
def gas_station(gas,cost):
    remaining=0
    candidate=0
    for i in range(len(gas)):
        remaining+=gas[i]-cost[i]
        if remaining<0:
            candidate=i+1
            remaining=0
    prev_remaining=sum(gas[:candidate])-sum(cost[:candidate])
    if candidate==len(gas) or remaining+prev_remaining<0:
        return -1
    else:
        return candidate

#courses graph
def dfs(graph,vertex,path,order,visited):
    path.append(vertex)
    for neighbor in graph[vertex]:
        if neighbor in path:
            return False
        if neighbor not in visited:
            visited.add(neighbor)
            if not dfs(graph,neighbor,path,order,visited):
               return False
    order.append(path.pop())
    return True
def top_sort(graph):
    visited=set()
    path=[]
    order=[]
    for vertex in graph:
        if vertex not in visited:
            visited.add(vertex)
            dfs(graph,vertex,path,order,visited)
    return order[::-1]

def course_schedule(n,prerequisites):
    graph=[[] for i in range(n)]
    for pre in prerequisites:
        graph[pre[1].append(pre[0])]
    visited=set()
    path=set()
    order=[]
    for course in range(n):
        if course not in visited:
            visited.add(course)
            if not dfs(graph,course,path,order,visited):
                return False
    return True

#Kth permutation
def kth_permutation(n,k):
    permutations=list(itertools.permutations(range(1,n+1)))
    return ''.join(map(str,permutations[k-1]))
#or
def kth_permutation(n, k):
    permutation = []
    unused = list(range(1, n + 1))
    fact = [1] * (n + 1)

    for i in range(1, n + 1):
        fact[i] = i * fact[i - 1]
    k -= 1 
    while n > 0:
        part_length = fact[n] // n
        i = k // part_length
        permutation.append(unused[i])
        unused.pop(i)
        n -= 1
        k %= part_length
    return "".join(map(str, permutation))  

#contains all string
def contains_all(freq1,freq2):
    for ch in freq2:
        if freq1[ch]<freq2[ch]:
            return False
    return True
